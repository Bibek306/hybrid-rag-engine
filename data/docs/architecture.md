# Core Architecture & Performance Design

## Ephemeral Caching Architecture
To reduce tail latency on heavy read endpoints, the platform implements a write-through Redis cache cluster. Cache entries use a Time-To-Live (TTL) of 300 seconds. When database mutations occur, cache invalidation events are published asynchronously over an Apache Kafka event bus. This pattern ensures eventual consistency while keeping 99th percentile response times below 25 milliseconds.

## Asynchronous Background Processing
Heavy computation tasks, such as generating customer PDF invoices or running batch embedding extractions, are offloaded from the web request-response cycle. Tasks are pushed to a Celery task queue backed by Redis broker. Multiple stateless worker containers pull jobs off the queue concurrently, preventing HTTP request timeouts for end users.

## Data Partitioning and Horizontal Scaling
Relational data is partitioned across shards using a consistent hashing algorithm on the primary tenant ID. This prevents hot spots where a single large client saturates disk I/O on a shared database instance. Read replicas handle reporting queries, while write operations are strictly directed to the primary master node.