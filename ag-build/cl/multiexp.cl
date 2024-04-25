/*
 * Same multiexp algorithm used in Bellman, with some modifications.
 * https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L174
 * The CPU version of multiexp parallelism is done by dividing the exponent
 * values into smaller windows, and then applying a sequence of rounds to each
 * window. The GPU kernel not only assigns a thread to each window but also
 * divides the bases into several groups which highly increases the number of
 * threads running in parallel for calculating a multiexp instance.
 */


DEVICE void POINT_multiexp_group(
  GLOBAL POINT_affine *bases,
  GLOBAL SCALAR_repr *exps,
  GLOBAL POINT_jacobian *buckets,
  uint tid,
  uint glen
)
{
  POINT_jacobian* t_bucket = &buckets[tid];
  *t_bucket = POINT_ZERO;
  
  for(uint i = 0; i < glen; i++) {
    uint ind = SCALAR_get_bit(exps[i], tid);
    if(ind) *t_bucket = POINT_add_mixed(*t_bucket, bases[i]);
  }
  
  BARRIER_LOCAL();
}

DEVICE void POINT_aggregate_group(
  GLOBAL POINT_jacobian *buckets,
  uint tid,
  uint num_windows_log2
)
{
  POINT_jacobian res = POINT_ZERO;
  for(uint height = 0; height < num_windows_log2; height++) {
    if (tid % (1 << (height + 1)) == 0) {
      res = buckets[tid];
      for(uint i = 0; i < (1 << height); i++) {
        res = POINT_double(res);
      }
      buckets[tid] = POINT_add(res, buckets[tid + (1 << height)]); // 8
    }
    BARRIER_LOCAL();
  }
}

KERNEL void POINT_multiexp(
    GLOBAL POINT_affine *bases,
    GLOBAL POINT_jacobian *results,
    GLOBAL SCALAR_repr *exps,
    GLOBAL POINT_jacobian *buckets,
    uint n,
    uint num_groups,
    uint num_windows,
    uint window_size,
    uint num_windows_log2) 
{

  // We have `num_windows` * `num_groups` threads per multiexp.
  const uint gid = GET_GLOBAL_ID();
  if(gid >= num_windows * num_groups) return;

  // POINT_jacobian* buckets = (POINT_jacobian*)cuda_shared;
  const uint group_thread_len = num_windows;
  const uint group_input_len = n / num_groups;

  const uint group_id = gid / group_thread_len;
  const uint local_thread_id = gid % group_thread_len;

  POINT_affine *group_bases = &bases[group_id * group_input_len];
  SCALAR_repr *group_exps = &exps[group_id * group_input_len];
  POINT_jacobian *group_buckets = &buckets[group_id * group_thread_len];

  POINT_multiexp_group(group_bases, group_exps, group_buckets, local_thread_id, group_input_len);


  // 245 ms

  POINT_aggregate_group(group_buckets, local_thread_id, num_windows_log2);

  if (local_thread_id == 0) {
    results[group_id] = group_buckets[0];
  }
  // 260 ms
}