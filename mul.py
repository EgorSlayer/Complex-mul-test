import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import time as time
import reikna.cluda as cluda
from reikna.fft import FFT
from numpy.fft import fftn, ifftn
Lx = 256
Ly = 256
Lz = 128

platforms = cl.get_platforms()
ctx = cl.Context(dev_type=cl.device_type.GPU, properties=[(cl.context_properties.PLATFORM, platforms[0])])
queue = cl.CommandQueue(ctx)
api = cluda.ocl_api()
thr = api.Thread(queue)

num = Lx*Ly*Lz
a = np.zeros((Lz,Ly,Lx)).astype(np.complex64)

fft = FFT(thr.empty_like(a)).compile(thr)

a = np.random.rand(Lz,Ly,Lx).astype(np.complex64)
a_buf = cl_array.to_device(queue, a)

b = np.random.rand(Lz,Ly,Lx).astype(np.complex64)
b_buf = cl_array.to_device(queue, b)

c = np.zeros((Lz,Ly,Lx)).astype(np.complex64)
c_buf = cl_array.to_device(queue, c)

r = np.zeros((Lz,Ly,Lx)).astype(np.float32)
r_buf = cl_array.to_device(queue, r)

coord_syst = '''
const int lx = get_global_size(2);
const int ly = get_global_size(1);
const int lz = get_global_size(0);

const int pl = lx * ly;
int x = get_global_id(2);
int y = get_global_id(1);
int z = get_global_id(0);
int i = x + y * lx + z * pl;
'''

program = thr.compile('''
#include <pyopencl-complex.h>
KERNEL void mul(
GLOBAL_MEM cfloat_t *a,
GLOBAL_MEM cfloat_t *b,
GLOBAL_MEM cfloat_t *c)
{
'''+coord_syst+'''
c[i] = cfloat_mul(a[i],b[i]);
}

KERNEL void real(
GLOBAL_MEM cfloat_t *c,
GLOBAL_MEM float *R)
{
'''+coord_syst+'''
R[i] = cfloat_real(c[i]);
}
''')



mul = program.mul
real = program.real


start_time = time.monotonic()

fft(a_buf,a_buf)
fft(b_buf,b_buf)
mul(a_buf, b_buf, c_buf, global_size=(Lz,Ly,Lx))
fft(c_buf,c_buf,inverse=1)
real(c_buf,r_buf, global_size=(Lz,Ly,Lx))
print('GPU seconds: ', time.monotonic() - start_time)
#print(final_array)
print('_____________________-')

start_time = time.monotonic()


res_a = fftn(a)
res_b = fftn(b)
res_c = res_a*res_b
res_c = ifftn(res_c)
res_cpu = np.real(res_c)
print('CPU seconds: ', time.monotonic() - start_time)
print('_____________________-')


res_gpu = r_buf.get()
print(res_gpu)
print('_____________________-')
print(res_cpu)
