// Defines to make the code work with both, CUDA and OpenCL
#ifdef __NVCC__
  #define DEVICE __device__
  #define GLOBAL
  #define KERNEL extern "C" __global__
  #define LOCAL
  #define CONSTANT __constant__

  #define GET_GLOBAL_ID() blockIdx.x * blockDim.x + threadIdx.x
  #define GET_GROUP_ID() blockIdx.x
  #define GET_LOCAL_ID() threadIdx.x
  #define GET_LOCAL_SIZE() blockDim.x
  #define BARRIER_LOCAL() __syncthreads()

  typedef unsigned char uchar;

  #define CUDA
#else // OpenCL
  #define DEVICE
  #define GLOBAL __global
  #define KERNEL __kernel
  #define LOCAL __local
  #define CONSTANT __constant

  #define GET_GLOBAL_ID() get_global_id(0)
  #define GET_GROUP_ID() get_group_id(0)
  #define GET_LOCAL_ID() get_local_id(0)
  #define GET_LOCAL_SIZE() get_local_size(0)
  #define BARRIER_LOCAL() barrier(CLK_LOCAL_MEM_FENCE)
#endif

#ifdef __NV_CL_C_VERSION
#define OPENCL_NVIDIA
#endif

#if defined(__WinterPark__) || defined(__BeaverCreek__) || defined(__Turks__) || \
    defined(__Caicos__) || defined(__Tahiti__) || defined(__Pitcairn__) || \
    defined(__Capeverde__) || defined(__Cayman__) || defined(__Barts__) || \
    defined(__Cypress__) || defined(__Juniper__) || defined(__Redwood__) || \
    defined(__Cedar__) || defined(__ATI_RV770__) || defined(__ATI_RV730__) || \
    defined(__ATI_RV710__) || defined(__Loveland__) || defined(__GPU__) || \
    defined(__Hawaii__)
#define AMD
#endif

// Returns a * b + c + d, puts the carry in d
DEVICE ulong mac_with_carry_64(ulong a, ulong b, ulong c, ulong *d) {
  #if defined(OPENCL_NVIDIA) || defined(CUDA)
    ulong lo, hi;
    asm("mad.lo.cc.u64 %0, %2, %3, %4;\r\n"
        "madc.hi.u64 %1, %2, %3, 0;\r\n"
        "add.cc.u64 %0, %0, %5;\r\n"
        "addc.u64 %1, %1, 0;\r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(b), "l"(c), "l"(*d));
    *d = hi;
    return lo;
  #else
    ulong lo = a * b + c;
    ulong hi = mad_hi(a, b, (ulong)(lo < c));
    a = lo;
    lo += *d;
    hi += (lo < a);
    *d = hi;
    return lo;
  #endif
}

// Returns a + b, puts the carry in d
DEVICE ulong add_with_carry_64(ulong a, ulong *b) {
  #if defined(OPENCL_NVIDIA) || defined(CUDA)
    ulong lo, hi;
    asm("add.cc.u64 %0, %2, %3;\r\n"
        "addc.u64 %1, 0, 0;\r\n"
        : "=l"(lo), "=l"(hi) : "l"(a), "l"(*b));
    *b = hi;
    return lo;
  #else
    ulong lo = a + *b;
    *b = lo < a;
    return lo;
  #endif
}

// Returns a * b + c + d, puts the carry in d
DEVICE uint mac_with_carry_32(uint a, uint b, uint c, uint *d) {
  ulong res = (ulong)a * b + c + *d;
  *d = res >> 32;
  return res;
}

// Returns a + b, puts the carry in b
DEVICE uint add_with_carry_32(uint a, uint *b) {
  #if defined(OPENCL_NVIDIA) || defined(CUDA)
    uint lo, hi;
    asm("add.cc.u32 %0, %2, %3;\r\n"
        "addc.u32 %1, 0, 0;\r\n"
        : "=r"(lo), "=r"(hi) : "r"(a), "r"(*b));
    *b = hi;
    return lo;
  #else
    uint lo = a + *b;
    *b = lo < a;
    return lo;
  #endif
}

// Reverse the given bits. It's used by the FFT kernel.
DEVICE uint bitreverse(uint n, uint bits) {
  uint r = 0;
  for(int i = 0; i < bits; i++) {
    r = (r << 1) | (n & 1);
    n >>= 1;
  }
  return r;
}

#ifdef CUDA
// CUDA doesn't support local buffers ("dynamic shared memory" in CUDA lingo) as function
// arguments, but only a single globally defined extern value. Use `uchar` so that it is always
// allocated by the number of bytes.
extern __shared__ uchar cuda_shared[];

typedef uint uint32_t;
typedef int  int32_t;
typedef uint limb;

DEVICE inline uint32_t add_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

DEVICE inline uint32_t addc_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

DEVICE inline uint32_t addc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}


DEVICE inline uint32_t madlo(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madlo_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madloc_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madloc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhi(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhi_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhic_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE inline uint32_t madhic(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  asm volatile ("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

typedef struct {
  int32_t _position;
} chain_t;

DEVICE inline
void chain_init(chain_t *c) {
  c->_position = 0;
}

DEVICE inline
uint32_t chain_add(chain_t *ch, uint32_t a, uint32_t b) {
  uint32_t r;

  ch->_position++;
  if(ch->_position==1)
    r=add_cc(a, b);
  else
    r=addc_cc(a, b);
  return r;
}

DEVICE inline
uint32_t chain_madlo(chain_t *ch, uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  ch->_position++;
  if(ch->_position==1)
    r=madlo_cc(a, b, c);
  else
    r=madloc_cc(a, b, c);
  return r;
}

DEVICE inline
uint32_t chain_madhi(chain_t *ch, uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;

  ch->_position++;
  if(ch->_position==1)
    r=madhi_cc(a, b, c);
  else
    r=madhic_cc(a, b, c);
  return r;
}
#endif
#define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__limb uint
#define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS 8
#define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMB_BITS 32
#define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__INV 4026531839
typedef struct { ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__limb val[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS]; } ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_;
typedef struct { ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__limb val[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS]; } ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__repr;
CONSTANT ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__ONE = { { 1342177275, 2895524892, 2673921321, 922515093, 2021213742, 1718526831, 2584207151, 235567041 } };
CONSTANT ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__P = { { 4026531841, 1138881939, 2042196113, 674490440, 2172737629, 3092268470, 3778125865, 811880050 } };
CONSTANT ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__R2 = { { 2921426343, 465102405, 3814480355, 1409170097, 1404797061, 2353627965, 2135835813, 35049649 } };
CONSTANT ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__ZERO = { { 0, 0, 0, 0, 0, 0, 0, 0 } };
#if defined(OPENCL_NVIDIA) || defined(CUDA)

DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__sub_nvidia(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ b) {
asm("sub.cc.u32 %0, %0, %8;\r\n"
"subc.cc.u32 %1, %1, %9;\r\n"
"subc.cc.u32 %2, %2, %10;\r\n"
"subc.cc.u32 %3, %3, %11;\r\n"
"subc.cc.u32 %4, %4, %12;\r\n"
"subc.cc.u32 %5, %5, %13;\r\n"
"subc.cc.u32 %6, %6, %14;\r\n"
"subc.u32 %7, %7, %15;\r\n"
:"+r"(a.val[0]), "+r"(a.val[1]), "+r"(a.val[2]), "+r"(a.val[3]), "+r"(a.val[4]), "+r"(a.val[5]), "+r"(a.val[6]), "+r"(a.val[7])
:"r"(b.val[0]), "r"(b.val[1]), "r"(b.val[2]), "r"(b.val[3]), "r"(b.val[4]), "r"(b.val[5]), "r"(b.val[6]), "r"(b.val[7]));
return a;
}
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__add_nvidia(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ b) {
asm("add.cc.u32 %0, %0, %8;\r\n"
"addc.cc.u32 %1, %1, %9;\r\n"
"addc.cc.u32 %2, %2, %10;\r\n"
"addc.cc.u32 %3, %3, %11;\r\n"
"addc.cc.u32 %4, %4, %12;\r\n"
"addc.cc.u32 %5, %5, %13;\r\n"
"addc.cc.u32 %6, %6, %14;\r\n"
"addc.u32 %7, %7, %15;\r\n"
:"+r"(a.val[0]), "+r"(a.val[1]), "+r"(a.val[2]), "+r"(a.val[3]), "+r"(a.val[4]), "+r"(a.val[5]), "+r"(a.val[6]), "+r"(a.val[7])
:"r"(b.val[0]), "r"(b.val[1]), "r"(b.val[2]), "r"(b.val[3]), "r"(b.val[4]), "r"(b.val[5]), "r"(b.val[6]), "r"(b.val[7]));
return a;
}
#endif

// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)

#define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__BITS (ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS * ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMB_BITS)
#if ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMB_BITS == 32
  #define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__mac_with_carry mac_with_carry_32
  #define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__add_with_carry add_with_carry_32
#elif ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMB_BITS == 64
  #define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__mac_with_carry mac_with_carry_64
  #define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__add_with_carry add_with_carry_64
#endif

// Greater than or equal
DEVICE bool ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__gte(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ b) {
  for(char i = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS - 1; i >= 0; i--){
    if(a.val[i] > b.val[i])
      return true;
    if(a.val[i] < b.val[i])
      return false;
  }
  return true;
}

// Equals
DEVICE bool ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__eq(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ b) {
  for(uchar i = 0; i < ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS; i++)
    if(a.val[i] != b.val[i])
      return false;
  return true;
}

// Normal addition
#if defined(OPENCL_NVIDIA) || defined(CUDA)
  #define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__add_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__add_nvidia
  #define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__sub_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__sub_nvidia
#else
  DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__add_(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ b) {
    bool carry = 0;
    for(uchar i = 0; i < ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS; i++) {
      ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__limb old = a.val[i];
      a.val[i] += b.val[i] + carry;
      carry = carry ? old >= a.val[i] : old > a.val[i];
    }
    return a;
  }
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__sub_(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ b) {
    bool borrow = 0;
    for(uchar i = 0; i < ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS; i++) {
      ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__limb old = a.val[i];
      a.val[i] -= b.val[i] + borrow;
      borrow = borrow ? old <= a.val[i] : old < a.val[i];
    }
    return a;
  }
#endif

// Modular subtraction
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__sub(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ b) {
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ res = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__sub_(a, b);
  if(!ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__gte(a, b)) res = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__add_(res, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__P);
  return res;
}

// Modular addition
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__add(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ b) {
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ res = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__add_(a, b);
  if(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__gte(res, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__P)) res = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__sub_(res, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__P);
  return res;
}


#ifdef CUDA
// Code based on the work from Supranational, with special thanks to Niall Emmart:
//
// We would like to acknowledge Niall Emmart at Nvidia for his significant
// contribution of concepts and code for generating efficient SASS on
// Nvidia GPUs. The following papers may be of interest:
//     Optimizing Modular Multiplication for NVIDIA's Maxwell GPUs
//     https://ieeexplore.ieee.org/document/7563271
//
//     Faster modular exponentiation using double precision floating point
//     arithmetic on the GPU
//     https://ieeexplore.ieee.org/document/8464792

DEVICE void ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__reduce(uint32_t accLow[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS], uint32_t np0, uint32_t fq[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS]) {
  // accLow is an IN and OUT vector
  // count must be even
  const uint32_t count = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS;
  uint32_t accHigh[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS];
  uint32_t bucket=0, lowCarry=0, highCarry=0, q;
  int32_t  i, j;

  #pragma unroll
  for(i=0;i<count;i++)
    accHigh[i]=0;

  // bucket is used so we don't have to push a carry all the way down the line

  #pragma unroll
  for(j=0;j<count;j++) {       // main iteration
    if(j%2==0) {
      add_cc(bucket, 0xFFFFFFFF);
      accLow[0]=addc_cc(accLow[0], accHigh[1]);
      bucket=addc(0, 0);

      q=accLow[0]*np0;

      chain_t chain1;
      chain_init(&chain1);

      #pragma unroll
      for(i=0;i<count;i+=2) {
        accLow[i]=chain_madlo(&chain1, q, fq[i], accLow[i]);
        accLow[i+1]=chain_madhi(&chain1, q, fq[i], accLow[i+1]);
      }
      lowCarry=chain_add(&chain1, 0, 0);

      chain_t chain2;
      chain_init(&chain2);
      for(i=0;i<count-2;i+=2) {
        accHigh[i]=chain_madlo(&chain2, q, fq[i+1], accHigh[i+2]);    // note the shift down
        accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], accHigh[i+3]);
      }
      accHigh[i]=chain_madlo(&chain2, q, fq[i+1], highCarry);
      accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], 0);
    }
    else {
      add_cc(bucket, 0xFFFFFFFF);
      accHigh[0]=addc_cc(accHigh[0], accLow[1]);
      bucket=addc(0, 0);

      q=accHigh[0]*np0;

      chain_t chain3;
      chain_init(&chain3);
      #pragma unroll
      for(i=0;i<count;i+=2) {
        accHigh[i]=chain_madlo(&chain3, q, fq[i], accHigh[i]);
        accHigh[i+1]=chain_madhi(&chain3, q, fq[i], accHigh[i+1]);
      }
      highCarry=chain_add(&chain3, 0, 0);

      chain_t chain4;
      chain_init(&chain4);
      for(i=0;i<count-2;i+=2) {
        accLow[i]=chain_madlo(&chain4, q, fq[i+1], accLow[i+2]);    // note the shift down
        accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], accLow[i+3]);
      }
      accLow[i]=chain_madlo(&chain4, q, fq[i+1], lowCarry);
      accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], 0);
    }
  }

  // at this point, accHigh needs to be shifted back a word and added to accLow
  // we'll use one other trick.  Bucket is either 0 or 1 at this point, so we
  // can just push it into the carry chain.

  chain_t chain5;
  chain_init(&chain5);
  chain_add(&chain5, bucket, 0xFFFFFFFF);    // push the carry into the chain
  #pragma unroll
  for(i=0;i<count-1;i++)
    accLow[i]=chain_add(&chain5, accLow[i], accHigh[i+1]);
  accLow[i]=chain_add(&chain5, accLow[i], highCarry);
}

// Requirement: yLimbs >= xLimbs
DEVICE inline
void ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__mult_v1(uint32_t *x, uint32_t *y, uint32_t *xy) {
  const uint32_t xLimbs  = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS;
  const uint32_t yLimbs  = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS;
  const uint32_t xyLimbs = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS * 2;
  uint32_t temp[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS * 2];
  uint32_t carry = 0;

  #pragma unroll
  for (int32_t i = 0; i < xyLimbs; i++) {
    temp[i] = 0;
  }

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain1;
    chain_init(&chain1);
    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 1) {
        temp[i + j - 1] = chain_madlo(&chain1, x[i], y[j], temp[i + j - 1]);
        temp[i + j]     = chain_madhi(&chain1, x[i], y[j], temp[i + j]);
      }
    }
    if (i % 2 == 1) {
      temp[i + yLimbs - 1] = chain_add(&chain1, 0, 0);
    }
  }

  #pragma unroll
  for (int32_t i = xyLimbs - 1; i > 0; i--) {
    temp[i] = temp[i - 1];
  }
  temp[0] = 0;

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain2;
    chain_init(&chain2);

    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 0) {
        temp[i + j]     = chain_madlo(&chain2, x[i], y[j], temp[i + j]);
        temp[i + j + 1] = chain_madhi(&chain2, x[i], y[j], temp[i + j + 1]);
      }
    }
    if ((i + yLimbs) % 2 == 0 && i != yLimbs - 1) {
      temp[i + yLimbs]     = chain_add(&chain2, temp[i + yLimbs], carry);
      temp[i + yLimbs + 1] = chain_add(&chain2, temp[i + yLimbs + 1], 0);
      carry = chain_add(&chain2, 0, 0);
    }
    if ((i + yLimbs) % 2 == 1 && i != yLimbs - 1) {
      carry = chain_add(&chain2, carry, 0);
    }
  }

  #pragma unroll
  for(int32_t i = 0; i < xyLimbs; i++) {
    xy[i] = temp[i];
  }
}

DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__mul_nvidia(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ b) {
  // Perform full multiply
  limb ab[2 * ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS];
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__mult_v1(a.val, b.val, ab);

  uint32_t io[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS];
  #pragma unroll
  for(int i=0;i<ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS;i++) {
    io[i]=ab[i];
  }
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__reduce(io, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__INV, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__P.val);

  // Add io to the upper words of ab
  ab[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS] = add_cc(ab[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS], io[0]);
  int j;
  #pragma unroll
  for (j = 1; j < ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS - 1; j++) {
    ab[j + ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS] = addc_cc(ab[j + ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS], io[j]);
  }
  ab[2 * ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS - 1] = addc(ab[2 * ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS - 1], io[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS - 1]);

  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ r;
  #pragma unroll
  for (int i = 0; i < ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS; i++) {
    r.val[i] = ab[i + ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS];
  }

  if (ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__gte(r, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__P)) {
    r = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__sub_(r, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__P);
  }

  return r;
}

#endif

// Modular multiplication
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__mul_default(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ b) {
  /* CIOS Montgomery multiplication, inspired from Tolga Acar's thesis:
   * https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
   * Learn more:
   * https://en.wikipedia.org/wiki/Montgomery_modular_multiplication
   * https://alicebob.cryptoland.net/understanding-the-montgomery-reduction-algorithm/
   */
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__limb t[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS + 2] = {0};
  for(uchar i = 0; i < ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS; i++) {
    ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__limb carry = 0;
    for(uchar j = 0; j < ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS; j++)
      t[j] = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__mac_with_carry(a.val[j], b.val[i], t[j], &carry);
    t[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS] = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__add_with_carry(t[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS], &carry);
    t[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS + 1] = carry;

    carry = 0;
    ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__limb m = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__INV * t[0];
    ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__mac_with_carry(m, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__P.val[0], t[0], &carry);
    for(uchar j = 1; j < ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS; j++)
      t[j - 1] = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__mac_with_carry(m, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__P.val[j], t[j], &carry);

    t[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS - 1] = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__add_with_carry(t[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS], &carry);
    t[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS] = t[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS + 1] + carry;
  }

  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ result;
  for(uchar i = 0; i < ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS; i++) result.val[i] = t[i];

  if(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__gte(result, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__P)) result = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__sub_(result, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__P);

  return result;
}

#ifdef CUDA
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__mul(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ b) {
  return ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__mul_nvidia(a, b);
}
#else
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__mul(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ b) {
  return ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__mul_default(a, b);
}
#endif

// Squaring is a special case of multiplication which can be done ~1.5x faster.
// https://stackoverflow.com/a/16388571/1348497
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__sqr(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ a) {
  return ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__mul(a, a);
}

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__add(a, a)
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__double(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ a) {
  for(uchar i = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS - 1; i >= 1; i--)
    a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> (ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMB_BITS - 1));
  a.val[0] <<= 1;
  if(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__gte(a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__P)) a = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__sub_(a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__P);
  return a;
}

// Modular exponentiation (Exponentiation by Squaring)
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__pow(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ base, uint exponent) {
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ res = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__ONE;
  while(exponent > 0) {
    if (exponent & 1)
      res = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__mul(res, base);
    exponent = exponent >> 1;
    base = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__sqr(base);
  }
  return res;
}


// Store squares of the base in a lookup table for faster evaluation.
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__pow_lookup(GLOBAL ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ *bases, uint exponent) {
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ res = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__ONE;
  uint i = 0;
  while(exponent > 0) {
    if (exponent & 1)
      res = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__mul(res, bases[i]);
    exponent = exponent >> 1;
    i++;
  }
  return res;
}


DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__mont(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__repr a) {
  #ifdef CUDA
    ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ input = reinterpret_cast<ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_&>(a);  
  #else
    ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ input = * (ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ *) &a;
  #endif

  return ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__mul(input, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__R2);
}

DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__repr ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__unmont(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ a) {
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ one = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__ZERO;
  one.val[0] = 1;
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ unmont = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__mul(a, one);

  
  #ifdef CUDA
    ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__repr answer = reinterpret_cast<ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__repr&>(unmont);  
  #else
    ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__repr answer = * (ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__repr *) &unmont;
  #endif
  return answer;
}

// Get `i`th bit (From most significant digit) of the field.
DEVICE bool ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__get_bit(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__repr l, uint i) {
  return (l.val[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMBS - 1 - i / ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMB_BITS] >> (ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMB_BITS - 1 - (i % ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__LIMB_BITS))) & 1;
}

// Get `window` consecutive bits, (Starting from `skip`th bit) from the field.
DEVICE uint ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__get_bits(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__repr l, uint skip, uint window) {
  uint ret = 0;
  for(uint i = 0; i < window; i++) {
    ret <<= 1;
    ret |= ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__get_bit(l, skip + i);
  }
  return ret;
}

#define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__limb uint
#define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS 8
#define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMB_BITS 32
#define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__INV 3834012553
typedef struct { ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__limb val[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS]; } ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_;
typedef struct { ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__limb val[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS]; } ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__repr;
CONSTANT ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__ONE = { { 3314486685, 3546104717, 4123462461, 175696680, 2021213740, 1718526831, 2584207151, 235567041 } };
CONSTANT ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__P = { { 3632069959, 1008765974, 1752287885, 2541841041, 2172737629, 3092268470, 3778125865, 811880050 } };
CONSTANT ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__R2 = { { 1401617033, 4079811675, 3561292283, 3051821329, 172064758, 1202396927, 3401069855, 114859889 } };
CONSTANT ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__ZERO = { { 0, 0, 0, 0, 0, 0, 0, 0 } };
#if defined(OPENCL_NVIDIA) || defined(CUDA)

DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub_nvidia(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ b) {
asm("sub.cc.u32 %0, %0, %8;\r\n"
"subc.cc.u32 %1, %1, %9;\r\n"
"subc.cc.u32 %2, %2, %10;\r\n"
"subc.cc.u32 %3, %3, %11;\r\n"
"subc.cc.u32 %4, %4, %12;\r\n"
"subc.cc.u32 %5, %5, %13;\r\n"
"subc.cc.u32 %6, %6, %14;\r\n"
"subc.u32 %7, %7, %15;\r\n"
:"+r"(a.val[0]), "+r"(a.val[1]), "+r"(a.val[2]), "+r"(a.val[3]), "+r"(a.val[4]), "+r"(a.val[5]), "+r"(a.val[6]), "+r"(a.val[7])
:"r"(b.val[0]), "r"(b.val[1]), "r"(b.val[2]), "r"(b.val[3]), "r"(b.val[4]), "r"(b.val[5]), "r"(b.val[6]), "r"(b.val[7]));
return a;
}
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__add_nvidia(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ b) {
asm("add.cc.u32 %0, %0, %8;\r\n"
"addc.cc.u32 %1, %1, %9;\r\n"
"addc.cc.u32 %2, %2, %10;\r\n"
"addc.cc.u32 %3, %3, %11;\r\n"
"addc.cc.u32 %4, %4, %12;\r\n"
"addc.cc.u32 %5, %5, %13;\r\n"
"addc.cc.u32 %6, %6, %14;\r\n"
"addc.u32 %7, %7, %15;\r\n"
:"+r"(a.val[0]), "+r"(a.val[1]), "+r"(a.val[2]), "+r"(a.val[3]), "+r"(a.val[4]), "+r"(a.val[5]), "+r"(a.val[6]), "+r"(a.val[7])
:"r"(b.val[0]), "r"(b.val[1]), "r"(b.val[2]), "r"(b.val[3]), "r"(b.val[4]), "r"(b.val[5]), "r"(b.val[6]), "r"(b.val[7]));
return a;
}
#endif

// FinalityLabs - 2019
// Arbitrary size prime-field arithmetic library (add, sub, mul, pow)

#define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__BITS (ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS * ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMB_BITS)
#if ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMB_BITS == 32
  #define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mac_with_carry mac_with_carry_32
  #define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__add_with_carry add_with_carry_32
#elif ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMB_BITS == 64
  #define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mac_with_carry mac_with_carry_64
  #define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__add_with_carry add_with_carry_64
#endif

// Greater than or equal
DEVICE bool ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__gte(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ b) {
  for(char i = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS - 1; i >= 0; i--){
    if(a.val[i] > b.val[i])
      return true;
    if(a.val[i] < b.val[i])
      return false;
  }
  return true;
}

// Equals
DEVICE bool ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__eq(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ b) {
  for(uchar i = 0; i < ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS; i++)
    if(a.val[i] != b.val[i])
      return false;
  return true;
}

// Normal addition
#if defined(OPENCL_NVIDIA) || defined(CUDA)
  #define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__add_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__add_nvidia
  #define ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub_nvidia
#else
  DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__add_(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ b) {
    bool carry = 0;
    for(uchar i = 0; i < ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS; i++) {
      ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__limb old = a.val[i];
      a.val[i] += b.val[i] + carry;
      carry = carry ? old >= a.val[i] : old > a.val[i];
    }
    return a;
  }
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub_(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ b) {
    bool borrow = 0;
    for(uchar i = 0; i < ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS; i++) {
      ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__limb old = a.val[i];
      a.val[i] -= b.val[i] + borrow;
      borrow = borrow ? old <= a.val[i] : old < a.val[i];
    }
    return a;
  }
#endif

// Modular subtraction
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ b) {
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ res = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub_(a, b);
  if(!ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__gte(a, b)) res = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__add_(res, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__P);
  return res;
}

// Modular addition
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__add(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ b) {
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ res = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__add_(a, b);
  if(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__gte(res, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__P)) res = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub_(res, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__P);
  return res;
}


#ifdef CUDA
// Code based on the work from Supranational, with special thanks to Niall Emmart:
//
// We would like to acknowledge Niall Emmart at Nvidia for his significant
// contribution of concepts and code for generating efficient SASS on
// Nvidia GPUs. The following papers may be of interest:
//     Optimizing Modular Multiplication for NVIDIA's Maxwell GPUs
//     https://ieeexplore.ieee.org/document/7563271
//
//     Faster modular exponentiation using double precision floating point
//     arithmetic on the GPU
//     https://ieeexplore.ieee.org/document/8464792

DEVICE void ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__reduce(uint32_t accLow[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS], uint32_t np0, uint32_t fq[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS]) {
  // accLow is an IN and OUT vector
  // count must be even
  const uint32_t count = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS;
  uint32_t accHigh[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS];
  uint32_t bucket=0, lowCarry=0, highCarry=0, q;
  int32_t  i, j;

  #pragma unroll
  for(i=0;i<count;i++)
    accHigh[i]=0;

  // bucket is used so we don't have to push a carry all the way down the line

  #pragma unroll
  for(j=0;j<count;j++) {       // main iteration
    if(j%2==0) {
      add_cc(bucket, 0xFFFFFFFF);
      accLow[0]=addc_cc(accLow[0], accHigh[1]);
      bucket=addc(0, 0);

      q=accLow[0]*np0;

      chain_t chain1;
      chain_init(&chain1);

      #pragma unroll
      for(i=0;i<count;i+=2) {
        accLow[i]=chain_madlo(&chain1, q, fq[i], accLow[i]);
        accLow[i+1]=chain_madhi(&chain1, q, fq[i], accLow[i+1]);
      }
      lowCarry=chain_add(&chain1, 0, 0);

      chain_t chain2;
      chain_init(&chain2);
      for(i=0;i<count-2;i+=2) {
        accHigh[i]=chain_madlo(&chain2, q, fq[i+1], accHigh[i+2]);    // note the shift down
        accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], accHigh[i+3]);
      }
      accHigh[i]=chain_madlo(&chain2, q, fq[i+1], highCarry);
      accHigh[i+1]=chain_madhi(&chain2, q, fq[i+1], 0);
    }
    else {
      add_cc(bucket, 0xFFFFFFFF);
      accHigh[0]=addc_cc(accHigh[0], accLow[1]);
      bucket=addc(0, 0);

      q=accHigh[0]*np0;

      chain_t chain3;
      chain_init(&chain3);
      #pragma unroll
      for(i=0;i<count;i+=2) {
        accHigh[i]=chain_madlo(&chain3, q, fq[i], accHigh[i]);
        accHigh[i+1]=chain_madhi(&chain3, q, fq[i], accHigh[i+1]);
      }
      highCarry=chain_add(&chain3, 0, 0);

      chain_t chain4;
      chain_init(&chain4);
      for(i=0;i<count-2;i+=2) {
        accLow[i]=chain_madlo(&chain4, q, fq[i+1], accLow[i+2]);    // note the shift down
        accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], accLow[i+3]);
      }
      accLow[i]=chain_madlo(&chain4, q, fq[i+1], lowCarry);
      accLow[i+1]=chain_madhi(&chain4, q, fq[i+1], 0);
    }
  }

  // at this point, accHigh needs to be shifted back a word and added to accLow
  // we'll use one other trick.  Bucket is either 0 or 1 at this point, so we
  // can just push it into the carry chain.

  chain_t chain5;
  chain_init(&chain5);
  chain_add(&chain5, bucket, 0xFFFFFFFF);    // push the carry into the chain
  #pragma unroll
  for(i=0;i<count-1;i++)
    accLow[i]=chain_add(&chain5, accLow[i], accHigh[i+1]);
  accLow[i]=chain_add(&chain5, accLow[i], highCarry);
}

// Requirement: yLimbs >= xLimbs
DEVICE inline
void ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mult_v1(uint32_t *x, uint32_t *y, uint32_t *xy) {
  const uint32_t xLimbs  = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS;
  const uint32_t yLimbs  = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS;
  const uint32_t xyLimbs = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS * 2;
  uint32_t temp[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS * 2];
  uint32_t carry = 0;

  #pragma unroll
  for (int32_t i = 0; i < xyLimbs; i++) {
    temp[i] = 0;
  }

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain1;
    chain_init(&chain1);
    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 1) {
        temp[i + j - 1] = chain_madlo(&chain1, x[i], y[j], temp[i + j - 1]);
        temp[i + j]     = chain_madhi(&chain1, x[i], y[j], temp[i + j]);
      }
    }
    if (i % 2 == 1) {
      temp[i + yLimbs - 1] = chain_add(&chain1, 0, 0);
    }
  }

  #pragma unroll
  for (int32_t i = xyLimbs - 1; i > 0; i--) {
    temp[i] = temp[i - 1];
  }
  temp[0] = 0;

  #pragma unroll
  for (int32_t i = 0; i < xLimbs; i++) {
    chain_t chain2;
    chain_init(&chain2);

    #pragma unroll
    for (int32_t j = 0; j < yLimbs; j++) {
      if ((i + j) % 2 == 0) {
        temp[i + j]     = chain_madlo(&chain2, x[i], y[j], temp[i + j]);
        temp[i + j + 1] = chain_madhi(&chain2, x[i], y[j], temp[i + j + 1]);
      }
    }
    if ((i + yLimbs) % 2 == 0 && i != yLimbs - 1) {
      temp[i + yLimbs]     = chain_add(&chain2, temp[i + yLimbs], carry);
      temp[i + yLimbs + 1] = chain_add(&chain2, temp[i + yLimbs + 1], 0);
      carry = chain_add(&chain2, 0, 0);
    }
    if ((i + yLimbs) % 2 == 1 && i != yLimbs - 1) {
      carry = chain_add(&chain2, carry, 0);
    }
  }

  #pragma unroll
  for(int32_t i = 0; i < xyLimbs; i++) {
    xy[i] = temp[i];
  }
}

DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul_nvidia(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ b) {
  // Perform full multiply
  limb ab[2 * ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS];
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mult_v1(a.val, b.val, ab);

  uint32_t io[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS];
  #pragma unroll
  for(int i=0;i<ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS;i++) {
    io[i]=ab[i];
  }
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__reduce(io, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__INV, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__P.val);

  // Add io to the upper words of ab
  ab[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS] = add_cc(ab[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS], io[0]);
  int j;
  #pragma unroll
  for (j = 1; j < ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS - 1; j++) {
    ab[j + ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS] = addc_cc(ab[j + ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS], io[j]);
  }
  ab[2 * ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS - 1] = addc(ab[2 * ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS - 1], io[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS - 1]);

  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ r;
  #pragma unroll
  for (int i = 0; i < ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS; i++) {
    r.val[i] = ab[i + ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS];
  }

  if (ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__gte(r, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__P)) {
    r = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub_(r, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__P);
  }

  return r;
}

#endif

// Modular multiplication
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul_default(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ b) {
  /* CIOS Montgomery multiplication, inspired from Tolga Acar's thesis:
   * https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
   * Learn more:
   * https://en.wikipedia.org/wiki/Montgomery_modular_multiplication
   * https://alicebob.cryptoland.net/understanding-the-montgomery-reduction-algorithm/
   */
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__limb t[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS + 2] = {0};
  for(uchar i = 0; i < ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS; i++) {
    ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__limb carry = 0;
    for(uchar j = 0; j < ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS; j++)
      t[j] = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mac_with_carry(a.val[j], b.val[i], t[j], &carry);
    t[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS] = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__add_with_carry(t[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS], &carry);
    t[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS + 1] = carry;

    carry = 0;
    ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__limb m = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__INV * t[0];
    ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mac_with_carry(m, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__P.val[0], t[0], &carry);
    for(uchar j = 1; j < ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS; j++)
      t[j - 1] = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mac_with_carry(m, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__P.val[j], t[j], &carry);

    t[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS - 1] = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__add_with_carry(t[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS], &carry);
    t[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS] = t[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS + 1] + carry;
  }

  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ result;
  for(uchar i = 0; i < ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS; i++) result.val[i] = t[i];

  if(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__gte(result, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__P)) result = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub_(result, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__P);

  return result;
}

#ifdef CUDA
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ b) {
  return ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul_nvidia(a, b);
}
#else
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ b) {
  return ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul_default(a, b);
}
#endif

// Squaring is a special case of multiplication which can be done ~1.5x faster.
// https://stackoverflow.com/a/16388571/1348497
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sqr(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ a) {
  return ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(a, a);
}

// Left-shift the limbs by one bit and subtract by modulus in case of overflow.
// Faster version of ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__add(a, a)
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__double(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ a) {
  for(uchar i = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS - 1; i >= 1; i--)
    a.val[i] = (a.val[i] << 1) | (a.val[i - 1] >> (ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMB_BITS - 1));
  a.val[0] <<= 1;
  if(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__gte(a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__P)) a = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub_(a, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__P);
  return a;
}

// Modular exponentiation (Exponentiation by Squaring)
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__pow(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ base, uint exponent) {
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ res = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__ONE;
  while(exponent > 0) {
    if (exponent & 1)
      res = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(res, base);
    exponent = exponent >> 1;
    base = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sqr(base);
  }
  return res;
}


// Store squares of the base in a lookup table for faster evaluation.
DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__pow_lookup(GLOBAL ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ *bases, uint exponent) {
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ res = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__ONE;
  uint i = 0;
  while(exponent > 0) {
    if (exponent & 1)
      res = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(res, bases[i]);
    exponent = exponent >> 1;
    i++;
  }
  return res;
}


DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mont(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__repr a) {
  #ifdef CUDA
    ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ input = reinterpret_cast<ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_&>(a);  
  #else
    ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ input = * (ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ *) &a;
  #endif

  return ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(input, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__R2);
}

DEVICE ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__repr ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__unmont(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ a) {
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ one = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__ZERO;
  one.val[0] = 1;
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ unmont = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(a, one);

  
  #ifdef CUDA
    ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__repr answer = reinterpret_cast<ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__repr&>(unmont);  
  #else
    ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__repr answer = * (ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__repr *) &unmont;
  #endif
  return answer;
}

// Get `i`th bit (From most significant digit) of the field.
DEVICE bool ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__get_bit(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__repr l, uint i) {
  return (l.val[ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMBS - 1 - i / ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMB_BITS] >> (ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMB_BITS - 1 - (i % ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__LIMB_BITS))) & 1;
}

// Get `window` consecutive bits, (Starting from `skip`th bit) from the field.
DEVICE uint ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__get_bits(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__repr l, uint skip, uint window) {
  uint ret = 0;
  for(uint i = 0; i < window; i++) {
    ret <<= 1;
    ret |= ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__get_bit(l, skip + i);
  }
  return ret;
}





// Elliptic curve operations (Short Weierstrass Jacobian form)

#define ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__ZERO ((ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian){ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__ZERO, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__ONE, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__ZERO})

typedef struct {
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ x;
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ y;
} ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__affine;

typedef struct {
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ x;
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ y;
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ z;
} ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian;

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
DEVICE ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__double(ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian inp) {
  const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ local_zero = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__ZERO;
  if(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__eq(inp.z, local_zero)) {
      return inp;
  }

  const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ a = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sqr(inp.x); // A = X1^2
  const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ b = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sqr(inp.y); // B = Y1^2
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ c = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sqr(b); // C = B^2

  // D = 2*((X1+B)2-A-C)
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ d = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__add(inp.x, b);
  d = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sqr(d); d = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(d, a), c); d = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__double(d);

  const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ e = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__add(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__double(a), a); // E = 3*A
  const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ f = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sqr(e);

  inp.z = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(inp.y, inp.z); inp.z = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__double(inp.z); // Z3 = 2*Y1*Z1
  inp.x = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(f, d), d); // X3 = F-2*D

  // Y3 = E*(D-X3)-8*C
  c = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__double(c); c = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__double(c); c = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__double(c);
  inp.y = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(d, inp.x), e), c);

  return inp;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
DEVICE ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__add_mixed(ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian a, ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__affine b) {
  const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ local_zero = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__ZERO;
  if(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__eq(a.z, local_zero)) {
    const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ local_one = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__ONE;
    a.x = b.x;
    a.y = b.y;
    a.z = local_one;
    return a;
  }

  const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ z1z1 = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sqr(a.z);
  const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ u2 = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(b.x, z1z1);
  const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ s2 = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(b.y, a.z), z1z1);

  if(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__eq(a.x, u2) && ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__eq(a.y, s2)) {
      return ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__double(a);
  }

  const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ h = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(u2, a.x); // H = U2-X1
  const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ hh = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sqr(h); // HH = H^2
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ i = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__double(hh); i = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__double(i); // I = 4*HH
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ j = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(h, i); // J = H*I
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ r = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(s2, a.y); r = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__double(r); // r = 2*(S2-Y1)
  const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ v = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(a.x, i);

  ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian ret;

  // X3 = r^2 - J - 2*V
  ret.x = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sqr(r), j), ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__double(v));

  // Y3 = r*(V-X3)-2*Y1*J
  j = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(a.y, j); j = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__double(j);
  ret.y = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(v, ret.x), r), j);

  // Z3 = (Z1+H)^2-Z1Z1-HH
  ret.z = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__add(a.z, h); ret.z = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sqr(ret.z), z1z1), hh);
  return ret;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
DEVICE ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__add(ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian a, ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian b) {

  const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ local_zero = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__ZERO;
  if(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__eq(a.z, local_zero)) return b;
  if(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__eq(b.z, local_zero)) return a;

  const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ z1z1 = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sqr(a.z); // Z1Z1 = Z1^2
  const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ z2z2 = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sqr(b.z); // Z2Z2 = Z2^2
  const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ u1 = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(a.x, z2z2); // U1 = X1*Z2Z2
  const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ u2 = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(b.x, z1z1); // U2 = X2*Z1Z1
  ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ s1 = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(a.y, b.z), z2z2); // S1 = Y1*Z2*Z2Z2
  const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ s2 = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(b.y, a.z), z1z1); // S2 = Y2*Z1*Z1Z1

  if(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__eq(u1, u2) && ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__eq(s1, s2))
    return ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__double(a);
  else {
    const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ h = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(u2, u1); // H = U2-U1
    ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ i = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__double(h); i = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sqr(i); // I = (2*H)^2
    const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ j = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(h, i); // J = H*I
    ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ r = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(s2, s1); r = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__double(r); // r = 2*(S2-S1)
    const ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4_ v = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(u1, i); // V = U1*I
    a.x = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sqr(r), j), v), v); // X3 = r^2 - J - 2*V

    // Y3 = r*(V - X3) - 2*S1*J
    a.y = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(v, a.x), r);
    s1 = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(s1, j); s1 = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__double(s1); // S1 = S1 * J * 2
    a.y = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(a.y, s1);

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    a.z = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__add(a.z, b.z); a.z = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sqr(a.z);
    a.z = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(a.z, z1z1), z2z2);
    a.z = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__mul(a.z, h);

    return a;
  }
}

DEVICE ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__neg(ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian a) {
  a.y = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__sub(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fq__FqConfig__4___4__ZERO, a.y);
  return a;
}

DEVICE ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__sub(ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian a, ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian b) {
  return ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__add(a, ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__neg(b));
}

DEVICE ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__mul_exponent(ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian base, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__repr exp) {
  ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian res = ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__ZERO;
  for(uint i = 0; i < ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__BITS; i++) {
    res = ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__double(res);
    bool exp_bit_i = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__get_bit(exp, i);
    if(exp_bit_i) res = ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__add(res, base);
  }
  return res;
}

DEVICE ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__mul(ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian base, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4_ exp) {
  return ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__mul_exponent(base, ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__unmont(exp));
}






/*
 * Same multiexp algorithm used in Bellman, with some modifications.
 * https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L174
 * The CPU version of multiexp parallelism is done by dividing the exponent
 * values into smaller windows, and then applying a sequence of rounds to each
 * window. The GPU kernel not only assigns a thread to each window but also
 * divides the bases into several groups which highly increases the number of
 * threads running in parallel for calculating a multiexp instance.
 */

KERNEL void ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__multiexp(
    GLOBAL ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__affine *bases,
    GLOBAL ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian *buckets,
    GLOBAL ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian *results,
    GLOBAL ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__repr *exps,
    uint n,
    uint num_groups,
    uint num_windows,
    uint window_size) {

  // We have `num_windows` * `num_groups` threads per multiexp.
  const uint gid = GET_GLOBAL_ID();
  if(gid >= num_windows * num_groups) return;

  // We have (2^window_size - 1) buckets.
  const uint bucket_len = ((1 << window_size) - 1);

  // Each thread has its own set of buckets in global memory.
  buckets += bucket_len * gid;

  const ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian local_zero = ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__ZERO;
  for(uint i = 0; i < bucket_len; i++) buckets[i] = local_zero;

  // Num of elements in each group. Round the number up (ceil).
  const uint len = (n + num_groups - 1) / num_groups;

  // This thread runs the multiexp algorithm on elements from `nstart` to `nend`
  // on the window [`bits`, `bits` + `w`)
  const uint nstart = len * (gid / num_windows);
  const uint nend = min(nstart + len, n);
  const uint bits = (gid % num_windows) * window_size;
  const ushort w = min((ushort)window_size, (ushort)(ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__BITS - bits));

  ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian res = ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__ZERO;
  for(uint i = nstart; i < nend; i++) {
    uint ind = ag_types__impls__ark_ff__fields__models__fp__Fp_ark_ff__fields__models__fp__montgomery_backend__MontBackend_ark_bn254__fields__fr__FrConfig__4___4__get_bits(exps[i], bits, w);

    #if defined(OPENCL_NVIDIA) || defined(CUDA)
      // O_o, weird optimization, having a single special case makes it
      // tremendously faster!
      // 511 is chosen because it's half of the maximum bucket len, but
      // any other number works... Bigger indices seems to be better...
      if(ind == 511) buckets[510] = ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__add_mixed(buckets[510], bases[i]);
      else if(ind--) buckets[ind] = ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__add_mixed(buckets[ind], bases[i]);
    #else
      if(ind--) buckets[ind] = ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__add_mixed(buckets[ind], bases[i]);
    #endif
  }

  // Summation by parts
  // e.g. 3a + 2b + 1c = a +
  //                    (a) + b +
  //                    ((a) + b) + c
  ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__jacobian acc = ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__ZERO;
  for(int j = bucket_len - 1; j >= 0; j--) {
    acc = ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__add(acc, buckets[j]);
    res = ag_types__impls__ark_ec__models__short_weierstrass__affine__Affine_ark_bn254__curves__g1__Config__add(res, acc);
  }

  results[gid] = res;
}





