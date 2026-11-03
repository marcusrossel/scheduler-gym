fn sched_init() -> Stats { Stats::default() }

fn sched_iter<'a, L: Language, N: Analysis<L>, IterData: IterationData<L, N>>(ctxt: &mut Ctxt<'a, L, N, IterData>, stats: &mut Stats) -> Result<(), StopReason> {
    let ex = Extractor::new(&ctxt.runner.egraph, AdditiveCostFn(ctxt.cfg.cf));
    let ctxt_cost = compute_ctxt_costs(&ex, ctxt);
    let mut classes: Vec<_> = ctxt_cost.iter().map(|(id, cost)| (*id, cost + ex.find_best_cost(*id))).collect();
    classes.sort_by_key(|x| x.1);
    classes.truncate(500);

    let mut matches = Vec::new();

    for (rw_i, rw) in ctxt.rws.iter().enumerate() {
        let mm = search_backoff_rewrite(stats, ctxt, rw, classes.iter().map(|(i, _)| *i))?;
        matches.push(mm);
        ctxt.check_limits()?;
    }

    for (rw, ms) in ctxt.rws.iter().zip(matches.into_iter()) {
        rw.apply(&mut ctxt.runner.egraph, &ms);
        ctxt.check_limits()?;
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

fn search_backoff_rewrite<'a, L: Language, N: Analysis<L>, IterData: IterationData<L, N>>(stats: &mut Stats, ctxt: &Ctxt<'_, L, N, IterData>, rw: &'a Rewrite<L, N>, classes: impl Iterator<Item=Id>) -> Result<Vec<SearchMatches<'a, L>>, StopReason> {
    let iteration = ctxt.runner.iterations.len();
    let egraph = &ctxt.runner.egraph;
    let stats = stats.entry(rw.name).or_insert(RuleStats {
        banned_until: 0,
        times_banned: 0,
        match_limit: 1000,
        ban_length: 5,
    });

    if iteration < stats.banned_until { return Ok(Vec::new()) }

    let threshold = stats
        .match_limit
        .checked_shl(stats.times_banned as u32)
        .unwrap();

    let mut matches = Vec::new();

    let mut total_len = 0;
    for c in classes {
        let local_threshold = (threshold - matches.len()).saturating_add(1);
        let submatches = rw.searcher.search_eclass_with_limit(&ctxt.runner.egraph, c, local_threshold);
        total_len += submatches.iter().map(|m| m.substs.len()).sum::<usize>();

        if total_len > threshold {
            let ban_length = stats.ban_length << stats.times_banned;
            stats.times_banned += 1;
            stats.banned_until = iteration + ban_length;
            return Ok(Vec::new())
        }

        matches.extend(submatches);
    }

    Ok(matches)
}

struct RuleStats {
    banned_until: usize,
    times_banned: usize,
    match_limit: usize,
    ban_length: usize,
}

type Stats = HashMap<Symbol, RuleStats>;

