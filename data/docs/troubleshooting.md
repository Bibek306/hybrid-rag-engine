# Incident Response & Troubleshooting Runbook

## GPU Memory Allocation Failures
Error code `CUDA_OUT_OF_MEMORY` or `torch.cuda.OutOfMemoryError` indicates that PyTorch attempted to allocate a tensor that exceeds available physical GPU VRAM. 
To resolve this:
1. Reduce training batch size by half (e.g., from 32 to 16).
2. Enable gradient accumulation steps to maintain effective batch size.
3. Call `torch.cuda.empty_cache()` between evaluation validation loops.
4. Use mixed-precision training (`torch.cuda.amp.autocast`) to convert FP32 weights to FP16 or BF16.

## Upstream Gateway Timeouts
An `HTTP 504 Gateway Timeout` indicates that an edge proxy (such as NGINX or AWS ALB) did not receive a timely response from the backend container worker.
Common root causes:
1. Long-running synchronous blocking calls in worker threads.
2. Database connection pool exhaustion under sudden traffic spikes.
3. The load balancer idle timeout is configured lower than the application worker timeout. Increase `proxy_read_timeout` in nginx configuration.

## Database Connection Refused
Error message `psycopg2.OperationalError: could not connect to server: Connection refused` means the PostgreSQL server is not accepting TCP/IP connections on port 5432.
Verify:
1. Check if the database service is running: `systemctl status postgresql`.
2. Inspect `postgresql.conf` to ensure `listen_addresses` includes `'*'`.
3. Check `pg_hba.conf` for client CIDR block authorization rules.