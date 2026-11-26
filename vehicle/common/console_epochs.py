"""Bounded long-running Redis for the console: quiescent epoch rotation.

Each epoch is a complete, independent run-owned Redis namespace with the
unchanged journal/codec/worker semantics (fixed stream IDs, fenced handoffs,
dedupe, XACK after confirmed output). The console only rotates when the epoch is
quiescent — no closure is published but unconfirmed, every consumer group has
zero pending and zero lag, every review is acknowledged — so no lifetime ever
spans two epochs and no unacknowledged event exists when an epoch is retired.
Event IDs are content-derived, so identities do not depend on the epoch.
"""
from dataclasses import dataclass, field

ROTATE_AT = 0.5  # rotate once any journal/stream budget is half used


@dataclass
class Epoch:
    index: int
    namespace: str
    workers: list
    review: object
    journal: object
    publisher: object
    review_published: dict = field(default_factory=dict)
    review_done: set = field(default_factory=set)
    releasing: set = field(default_factory=set)
    lifetimes: int = 0


def namespace_for(run_id, index):
    base = 'console-' + run_id.replace('-', '')[:24]
    return base if index == 0 else f'{base}-e{index}'


def open_epoch(index, *, client, codec, org, run_id, engine, profile):
    from .redis_scalar.worker import Worker
    from .redis_scalar.journal import Journal
    from .plate_requests.worker import RequestWorker
    from .plate_requests.host import V2Publisher
    from .manufacturer.streams import ReviewCodec, ReviewWorker
    namespace = namespace_for(run_id, index)
    workers = [Worker(client, codec=codec.v1.v0, namespace=namespace, role='colour', consumer='console-colour', **profile),
               Worker(client, codec=codec.v1, namespace=namespace, role='aggregator_plate', consumer='console-aggregator', **profile),
               RequestWorker(client, codec=codec, namespace=namespace, consumer='console-plate', **profile)]
    review = ReviewWorker(client, codec=ReviewCodec(codec.v1.v0), namespace=namespace, consumer='console-make', **profile)
    workers.append(review)
    for worker in workers:
        worker.setup()
    journal = Journal(client, namespace=namespace, trusted_org=org, role='plate_requests', consumer='console-engine', **profile)
    publisher = V2Publisher(client, journal, codec, engine)
    return Epoch(index, namespace, workers, review, journal, publisher)


def keys(client, epoch):
    return [k for k in client.scan_iter(match=epoch.namespace + ':*', count=1000)]


def usage(client, epoch, profile):
    """Largest used fraction of any journal event/vehicle budget or stream length."""
    worst = 0.0
    for journal in [w.journal for w in epoch.workers] + [epoch.journal]:
        values = client.hmget(journal.state, 'events', 'vehicles')
        events, vehicles = (int(v) if v is not None else 0 for v in values)
        worst = max(worst, events / profile['max_events'], vehicles / profile['max_vehicles'])
    for key in keys(client, epoch):
        if client.type(key) in (b'stream', 'stream'):
            worst = max(worst, client.xlen(key) / profile['max_stream'])
    return worst


def quiescent(client, epoch, engine, metadata_turn, *, turns=64):
    """True only when nothing of this epoch is unconfirmed, unread or unacknowledged."""
    if engine.pending or epoch.publisher.plans or epoch.publisher.writer.pending or epoch.releasing:
        return False
    if len(epoch.review_done) != len(epoch.review_published):
        return False
    for _ in range(turns):
        if not any([metadata_turn(worker) for worker in epoch.workers]):
            break
    for key in keys(client, epoch):
        if client.type(key) not in (b'stream', 'stream'):
            continue
        for group in client.xinfo_groups(key):
            lag = group.get('lag', group.get(b'lag'))
            pending = group.get('pending', group.get(b'pending'))
            if pending or lag:
                return False
    return True


def retire(client, epoch):
    """Release leases and delete every key of a quiescent, fully delivered epoch."""
    for worker in epoch.workers:
        if worker.journal.token:
            worker.journal.release()
    names = keys(client, epoch)
    for start in range(0, len(names), 500):
        client.delete(*names[start:start + 500])
    return len(names)
