from lcode2dPy.push_solver import PusherAndSolver
from lcode2dPy.beam.beam_slice import BeamSlice
from lcode2dPy.beam.beam_io import MemoryBeamSource, MemoryBeamDrain
#from lcode2dPy.push_solver_3d import PushAndSolver3d
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import imageio
from pathlib import Path
from lcode2dPy.config.default_config import default_config
from lcode2dPy.beam.beam_generator import make_beam, Gauss, rGauss
from lcode2dPy.plasma.initialization import init_plasma

class Simulation:
    def __init__(self, config=default_config, beam_generator=make_beam, beam_pars=None, diagnostics=None):
        self.config = config
        if config.get('geometry') == '3d':
            pass
#            self.push_solver = PushAndSolver3d(self.config) # 3d
        elif config.get('geometry') == 'circ' or config.get('geometry') == 'c':
            self.push_solver = PusherAndSolver(self.config) # circ
        elif config.get('geometry') == 'plane':
            self.push_solver = PusherAndSolver(self.config) # 2d_plane

        self.beam_generator = beam_generator
        self.beam_pars = beam_pars

        self.current_time = 0.
        self.beam_source = None
        self.beam_drain = None
        
        self.diagnostics = MyDiagnostics(config, diagnostics)

    def step(self, N_steps):
        # t step function, makes N_steps time steps.
        # Beam generation
        self.diagnostics.config()
        if self.beam_source is None:
            beam_particles = self.beam_generator(self.config, **self.beam_pars)
            beam_particle_dtype = np.dtype([('xi', 'f8'), ('r', 'f8'), ('p_z', 'f8'), ('p_r', 'f8'), ('M', 'f8'), ('q_m', 'f8'),
                               ('q_norm', 'f8'), ('id', 'i8')])
            beam_particles = np.array(list(map(tuple, beam_particles.to_numpy())), dtype=beam_particle_dtype)

            beam_slice = BeamSlice(beam_particles.size, beam_particles)
            self.beam_source = MemoryBeamSource(beam_slice) #TODO mpi_beam_source
            self.beam_drain = MemoryBeamDrain()
        # Time loop
        for t_i in range(N_steps):
            
            fields, plasma_particles = init_plasma(self.config)
            plasma_particles_new, fields_new = self.push_solver.step_dt(plasma_particles, fields, self.beam_source, self.beam_drain, self.current_time, self.diagnostics)
            beam_particles = self.beam_drain.beam_slice()
            beam_slice = BeamSlice(beam_particles.size, beam_particles)
            self.beam_source = MemoryBeamSource(beam_slice)
            self.beam_drain = MemoryBeamDrain()
            self.current_time = self.current_time + self.config.getfloat('time-step')
            print(f"\r[{'#'*int(t_i/(N_steps-1)*30)+' '*(30-int(t_i/(N_steps-1)*30))}] t={self.current_time}", end='')
            
            

# class Diagnostics2d:
#     def __init__(self, dt_diag, dxi_diag):
#         self.config = None
#         self.dt_diag = dt_diag
#         self.dxi_diag = dxi_diag

#     def every_dt(self):
#         pass # TODO

#     def every_dxi(self, t, layer_idx, plasma_particles, plasma_fields, rho_beam, beam_slice):
#         for diag_name in self.dxi_diag.keys():
#             diag, pars = self.dxi_diag[diag_name]
#             diag(self, t, layer_idx, plasma_particles, plasma_fields, rho_beam, beam_slice, **pars)
#         return None

    
class MyDiagnostics:
    def __init__(self, config, diag_list):
        self.c = config 
        self.diag_list = diag_list

    def config(self):
        for diag in self.diag_list:
            try:
                diag.config(self.c)
            except AttributeError:
                pass
    
    def dxi(self,*param):
        for diag in self.diag_list:
            diag.dxi(*param)

    def dump(self):
        for diag in self.diag_list:
            diag.dump()


class FieldDiagnostics:
    def __init__(self, name, r=0, 
                 t_start=None, t_end=None, period=100, 
                 cl_mem = False, 
                 out = 'i f',img_format = 'png', is_merge = False,
                 make_gif = False, gif_name=None):
        if name not in ['E_z', 'E_f', 'E_r', 'B_f', 'B_z']:
            raise AttributeError("Name isn't corrected")
        self.name = name
        self.r = r
        self.t_start = t_start 
        self.t_end = t_end
        self.period = period
        self.is_merge = is_merge
        self.out = out
        self.data = {}


    def config(self, config):
        self.dt = config.getfloat('time-step')
        self.tlim = config.getfloat('time-limit')
        if self.t_start is None:
            self.t_start = 0
        if self.t_end is None:
            self.t_end = self.tlim
        r_step = config.getfloat('r-step')
        xi_step = config.getfloat('xi-step')
        self.w = int(config.getfloat('window-length')/xi_step)
        self.h = int(config.getfloat('window-width')/r_step)+1
        
        self.last_idx = self.w-1
        
    
    
    def dxi(self, t, layer_idx, \
                plasma_particles, plasma_fields, rho_beam, \
                beam_slice):
        
        if t<self.t_start or t>self.t_end:
            return
        if (t-self.t_start)%self.period == 0:
            if layer_idx == 0:
                if self.r is None:
                    self.data[t] = np.empty((self.w, self.h)) # Grid
                else:
                    self.data[t] = np.empty(self.w)  # Line
            if self.r is None:
                self.data[t][layer_idx] = getattr(plasma_fields, self.name) 
            else:
                self.data[t][layer_idx] = getattr(plasma_fields, self.name)[self.r]
            
            if layer_idx == self.last_idx:
                self.dump(t)
                
    
    def dump(self,t):
        Path('./diagnostics/fields').mkdir(parents=True, exist_ok=True)
        if 'i' in self.out:
            if self.r is None:
                plt.imshow(self.data[t].T)
                plt.savefig(f'./diagnostics/fields/{self.name}_grid_{100*t:08.0f}.png')
                plt.close()
            else:
                plt.plot(self.data[t])
                plt.savefig(f'./diagnostics/fields/{self.name}_{100*t:08.0f}.png')
                plt.close()
        
        if 'f' in self.out:
            if self.r is None:
                np.save(f'./diagnostics/fields/{self.name}_grid_{100*t:08.0f}.npy', self.data)
            else:
                np.save(f'./diagnostics/fields/{self.name}_{100*t:08.0f}.npy',self.data)
    
    def make_gif(self, name=None):
        if name is None:
            name = self.name
        r_path = './gif_tmp/'
        path = Path(r_path)
        try:
            path.mkdir()
        except FileExistsError:
            shutil.rmtree(path)
            path.mkdir()
        files = []
        for key in self.data.keys():
            if self.r is None:
                plt.imshow(self.data[key].T)
            else:
                plt.ylim(-0.06,0.06) # magic
                plt.plot(self.data[key])
                
            file = r_path+str(key)+'.png'
            files.append(file)
            plt.savefig(file)
            plt.close()
        with imageio.get_writer(name+'.gif', mode='I') as writer:
            for filename in files:
                image = imageio.imread(filename)
                writer.append_data(image)
            for i in range(10):
                writer.append_data(image)
        shutil.rmtree(path)
        
            