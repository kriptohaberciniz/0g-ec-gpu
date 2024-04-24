/*
 * Same multiexp algorithm used in Bellman, with some modifications.
 * https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L174
 * The CPU version of multiexp parallelism is done by dividing the exponent
 * values into smaller windows, and then applying a sequence of rounds to each
 * window. The GPU kernel not only assigns a thread to each window but also
 * divides the bases into several groups which highly increases the number of
 * threads running in parallel for calculating a multiexp instance.
 */

KERNEL void POINT_multiexp(
    GLOBAL POINT_affine *bases,
    GLOBAL POINT_jacobian *results,
    GLOBAL SCALAR_repr *exps,
    uint n,
    uint num_groups,
    uint num_windows,
    uint window_size,
    uint num_windows_log2) 
{
  POINT_jacobian* buckets = (POINT_jacobian*)cuda_shared;

  // We have `num_windows` * `num_groups` threads per multiexp.
  const uint gid = GET_GLOBAL_ID();
  if(gid >= num_windows * num_groups) return;

  POINT_jacobian local_bucket = POINT_ZERO;

  // Num of elements in each group. Round the number up (ceil).
  const uint len = (n + num_groups - 1) / num_groups;

  // This thread runs the multiexp algorithm on elements from `nstart` to `nend`
  // on the window [`bits`, `bits` + `w`)
  const uint task_id = gid / num_windows;
  const uint nstart = len * task_id;
  const uint nend = min(nstart + len, n);
  const uint bits = (gid % num_windows) * window_size;
  
  for(uint i = nstart; i < nend; i++) {
    uint ind = SCALAR_get_bit(exps[i], bits);
    if(ind) local_bucket = POINT_add_mixed(local_bucket, bases[i]);
  }

  buckets[bits] = local_bucket;

  BARRIER_LOCAL();

  // 365 ms
  //if (bits == 0) {
  //  POINT_jacobian res = buckets[0]; 
  //  for(uint i = 1; i < num_windows; i++) {
  //    res = POINT_double(res); // 255
  //    res = POINT_add(res, buckets[i]); // 255
  //  }
  //  results[task_id] = res;
  //}

  // 280 ms
  uint step_2 = 1;
  uint step_2_new = 2;
  POINT_jacobian res = POINT_ZERO;
  for(uint step = 0; step < num_windows_log2; step++) {
    if (bits % step_2_new == 0) {
      res = buckets[bits];
      for(uint i = 0; i < step_2; i++) {
        res = POINT_double(res); // 1, 2, ..., 128 -> sum -> 255
      }
      buckets[bits] = POINT_add(res, buckets[bits + step_2]); // 8
    }
    step_2 = step_2_new;
    step_2_new <<= 1;
    BARRIER_LOCAL();
  }
  if (bits == 0) {
    results[task_id] = buckets[0];
  }
}
