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
  uint glen,
  uint num_thread_buckets,
  uint group_threads,
  uint window_bits
)
{
  const ushort w = min((ushort)window_bits, (ushort)(SCALAR_BITS - tid * window_bits));
  POINT_jacobian* t_buckets = &buckets[tid * num_thread_buckets];

  for(uint i = 0; i < num_thread_buckets; i++) {
    t_buckets[i] = POINT_ZERO;
  }
  
  for(uint i = 0; i < glen; i++) {
    uint ind = SCALAR_get_bits(exps[i], tid * window_bits, w);
    if (ind > 0) {
      POINT_jacobian* bucket = &t_buckets[ind - 1];
      *bucket = POINT_add_mixed(*bucket, bases[i]);
    }
  }

  POINT_jacobian acc = t_buckets[num_thread_buckets - 1];
  POINT_jacobian res = acc;
  for(int j = num_thread_buckets - 1; j >= 1; j--) {
    acc = POINT_add(acc, t_buckets[j - 1]);
    res = POINT_add(res, acc);
  }

  t_buckets[0] = res;
  
  BARRIER_LOCAL();
}

DEVICE void POINT_aggregate_group(
  GLOBAL POINT_jacobian *buckets,
  uint tid,
  uint num_thread_buckets,
  uint window_bits,
  uint height
)
{
  for(uint h = 0; h < height; h++) {
    uint lead_id = tid >> (h + 1) << (h + 1);
    uint sib_id = lead_id + (1 << h);
    uint bit_offset = (1 << h) * window_bits;

    if (tid != lead_id) {
      return;
    }

    POINT_jacobian res = buckets[lead_id * num_thread_buckets];
    for(uint i = 0; i < bit_offset; i++) {
      res = POINT_double(res);
    }
    buckets[lead_id * num_thread_buckets] = POINT_add(res, buckets[sib_id * num_thread_buckets]); // 8
  
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
    uint window_bits,
    uint num_windows_log2) 
{

  // We have `num_windows` * `num_groups` threads per multiexp.
  const uint gid = GET_GLOBAL_ID();
  if(gid >= num_windows * num_groups) return;

  // POINT_jacobian* buckets = (POINT_jacobian*)cuda_shared;
  const uint group_thread_len = (SCALAR_BITS + window_bits - 1) / window_bits;
  const uint group_input_len = n / num_groups;

  const uint group_id = gid / group_thread_len;
  const uint local_thread_id = gid % group_thread_len;

  const uint thread_buckets = (1 << window_bits) - 1;

  POINT_affine *group_bases = &bases[group_id * group_input_len];
  SCALAR_repr *group_exps = &exps[group_id * group_input_len];
  POINT_jacobian *group_buckets = &buckets[group_id * group_thread_len * thread_buckets];

  POINT_multiexp_group(group_bases, group_exps, group_buckets, local_thread_id, group_input_len, thread_buckets, group_thread_len, window_bits);


  // 245 ms

  POINT_aggregate_group(group_buckets, local_thread_id, thread_buckets, window_bits, num_windows_log2);

  if (local_thread_id == 0) {
    results[group_id] = group_buckets[0];
  }
  // 260 ms
}