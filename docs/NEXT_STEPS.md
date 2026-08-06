# Next steps

**Status: none of this is started.** Written 2026-08-05, recording judgement while it is fresh.
Nothing here is scheduled. Last commit before this doc was 2025-02-20.

---

## 1. The README is two sentences for 14,671 lines of Python

This is the highest-leverage fix in the repo and it is not close.

40 stars and 11 forks means people found this, read two sentences, and left. There is a
complete architecture in here — `client/`, `data/`, `env/`, `meta/`, `model/`, `trade/`,
`train/` — and none of it is described anywhere. Nobody can tell from the outside whether this
is a weekend toy or a real framework, so they assume the former.

What it needs, in order:

- One diagram of the pipeline: market data → environment → policy → agent → Alpaca execution.
- A runnable quickstart that trains something small and prints a result, with expected output.
- What `meta/pipe.py` and `meta/agent.py` actually do, since those are the interesting parts and
  the names give nothing away.
- An explicit statement of what is *not* done, so the first serious reader trusts the rest.

## 2. Dependency rot

`environment.yml` pins Python 3.10 and a conda-only install. `alpaca-py` is now at 0.43.5 and
has had breaking API changes since this was last touched. A first-time installer today most
likely hits an error before reaching a single line of this code.

- Move to `pyproject.toml` with `uv`; keep `environment.yml` only if conda users are a real
  audience, which they probably are not.
- Pin or bound `alpaca-py`, then actually run the quickstart against a paper account.
- Add CI that installs clean on 3.11–3.13 and runs the tests. There is a `tests/` directory
  and no evidence anything runs it.

## 3. Packaging

`neural` is taken on PyPI. `neural-trading` and `neural-rl` are both free as of 2026-08-05.
Without a package there is no `pip install`, and without that the star count stays where it is.

## 4. The honesty gap — the actually interesting one

Deep RL is the most overfitting-prone method in quantitative finance, by a wide margin. A
supervised factor backtest tries tens of specifications. A DRL run tries *seeds × learning rates
× network widths × reward shapings × episode counts* — the true trial count reaches the
thousands before anyone notices.

No DRL trading repository reports a multiple-testing-corrected Sharpe. Every one of them
reports the best run.

[Vintage](https://github.com/RezaSoleymanifar/vintage) implements the Deflated Sharpe Ratio
(Bailey & López de Prado, 2014) with a trial ledger. Wiring it in so `train/` logs every run
into that ledger, and evaluation reports a deflated Sharpe against the full sweep rather than
a raw Sharpe against the winner, would make this the only DRL trading repo that reports an
honest number.

That is a genuine differentiator and it is a small amount of code.

## 5. Point-in-time training data

`data/alpaca.py` is the only feed. Two problems for anything published:

- Adjusted prices are adjusted *retroactively*, so a model trained on today's adjusted history
  has seen split and dividend information that did not exist at the time.
- Alpaca's free historical tier does not reach far enough back for a convincing sample.

Vintage supplies a `known_at`-indexed panel that makes the leak structurally impossible, and
reaches further back through Yahoo. Worth doing before publishing any performance claim, and
not worth doing before §1 and §2, since nobody can run the thing yet.

---

## Not doing

- **Live capital.** Nothing here is validated well enough for that, and §4 exists precisely
  because the current numbers cannot be trusted.
- **Competing with LEAN or Nautilus Trader on execution realism.** Different problem.

## Order

§1 and §2 first — they are cheap and everything else is invisible without them. §4 is the one
worth being loud about once anyone can actually install this.
