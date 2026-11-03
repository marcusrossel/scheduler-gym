fn sched_init() -> BackoffScheduler { BackoffScheduler::default() }

fn sched_iter<'a, L: Language, N: Analysis<L>, IterData: IterationData<L, N>>(ctxt: &mut Ctxt<'a, L, N, IterData>, sched: &mut BackoffScheduler) -> Result<(), StopReason> {
    let ex = Extractor::new(&ctxt.runner.egraph, AdditiveCostFn(ctxt.cfg.cf));
    let ctxt_cost = compute_ctxt_costs(&ex, ctxt);

    let mut matches: BTreeMap</*detour cost*/ Cost, Vec<(usize, Id, Subst)>> = BTreeMap::default();

    for (rw_i, rw) in ctxt.rws.iter().enumerate() {
        let effective_pat = rw.searcher.get_pattern_ast().unwrap();

        for m in sched.search_rewrite(ctxt.runner.iterations.len(), &ctxt.runner.egraph, rw) {
            ctxt.check_limits()?;

            let lhs = m.eclass;
            for subst in m.substs {
                let class_cost = ex.find_best_cost(lhs);
                let cx_cost = *ctxt_cost.get(&lhs).unwrap_or(&ctxt.cfg.unreachable_cost); // this is the cost you get from not being able to reach any root.
                let detour_cost = cx_cost + class_cost;
                matches.entry(detour_cost).or_insert(Vec::new()).push((rw_i, lhs, subst));

                ctxt.check_limits()?;
            }
        }
    }

    let eg_data = |eg: &EGraph<_, _>| (eg.number_of_classes(), eg.total_size());

    let mut counter = 0;

    'outer: for (full_cost, new_apps) in matches {
        for (rw_i, lhs, subst) in &new_apps {
            let rw = &ctxt.rws[*rw_i];
            let pat_ast = rw.searcher.get_pattern_ast();

            let prev_data = eg_data(&ctxt.runner.egraph);
            rw.applier.apply_one(&mut ctxt.runner.egraph, *lhs, subst, pat_ast, rw.name);
            let post_data = eg_data(&ctxt.runner.egraph);

            if prev_data != post_data {
                counter += 1;
                if counter >= 400 { break 'outer }
            }

            ctxt.check_limits()?;
        }
    }

    ctxt.runner.egraph.rebuild();
    Ok(())
}

// === ctxt cost ===

fn compute_ctxt_costs<'a, L: Language, N: Analysis<L>, IterData: IterationData<L, N>>(ex: &Extractor<AdditiveCostFn<L>, L, N>, ctxt: &Ctxt<'a, L, N, IterData>) -> HashMap<Id, Cost> {
    let mut ctxt_cost = HashMap::new();

    let mut queue: MinPrioQueue<Cost, Id> = MinPrioQueue::new();

    // initial
    for root in &ctxt.runner.roots {
        queue.push(0, *root);
    }

    while let Some((cst, i)) = queue.pop() {
        if ctxt_cost.contains_key(&i) { continue }
        ctxt_cost.insert(i, cst);
        for e in &ctxt.runner.egraph[i].nodes {
            let e_cost = AdditiveCostFn(ctxt.cfg.cf).cost(e, |k| ex.find_best_cost(k));
            for &c in e.children() {
                // optimization: don't push junk to the queue.
                // NOTE: if we remembered what's the best thing we already pushed to the queue for some class,
                // we could do more efficient pruning.
                if ctxt_cost.contains_key(&c) { continue }

                let c_cost = ex.find_best_cost(c);
                let ncst = e_cost + cst - c_cost;
                queue.push(ncst, c);
            }
        }
    }

    ctxt_cost
}

fn pat_cost<L: Language, N: Analysis<L>>(pat: &PatternAst<L>, subst: &Subst, ex: &Extractor<AdditiveCostFn<L>, L, N>, cf: fn(&L) -> Cost) -> Cost {
    let mut vec: Vec<Cost> = Vec::new();
    for i in 0..pat.as_ref().len() {
        let cost = match &pat[i.into()] {
            ENodeOrVar::ENode(n) => AdditiveCostFn(cf).cost(n, |x| vec[usize::from(x)]),
            ENodeOrVar::Var(v) => ex.find_best_cost(subst[*v]),
        };
        vec.push(cost);
    }
    vec.last().copied().unwrap()
}

// === minqueue ===

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, BTreeMap};

struct MinPrioQueue<U, T>(BinaryHeap<WithOrdRev<U, T>>);

impl<U: Ord, T: Eq> MinPrioQueue<U, T> {
    pub fn new() -> Self {
        MinPrioQueue(BinaryHeap::default())
    }

    pub fn push(&mut self, u: U, t: T) {
        self.0.push(WithOrdRev(u, t));
    }

    pub fn pop(&mut self) -> Option<(U, T)> {
        self.0.pop().map(|WithOrdRev(u, t)| (u, t))
    }
}

// Takes the `Ord` from U, but reverses it.
#[derive(PartialEq, Eq, Debug)]
struct WithOrdRev<U, T>(pub U, pub T);

impl<U: Ord, T: Eq> PartialOrd for WithOrdRev<U, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // It's the other way around, because we want a min-heap!
        other.0.partial_cmp(&self.0)
    }
}
impl<U: Ord, T: Eq> Ord for WithOrdRev<U, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(&other).unwrap()
    }
}
