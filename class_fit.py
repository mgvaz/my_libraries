import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.interpolate import interp1d, LSQUnivariateSpline, UnivariateSpline, PPoly, splrep, splev, splder
from scipy.optimize import minimize,minimize_scalar, root_scalar

from IPython import display
import ipywidgets

from ipfnpytools.spline import constrained_spline

import os
from os import listdir
from os.path import isfile, join

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    directory=dir_name
    return directory

def get_data(pulse, interp=True, folder='r_ne_Vperp_inter/'):

    if interp==True:
        data = pd.read_csv(folder + f"r_ne_Vperp_{pulse}_inter.txt", sep="\s+", names=["R (m)", "$n_{e}$ ($10^{19}m^{-3}$)", "$v_{\perp}$ (km/s)"])
    if interp==False:
        data = pd.read_csv(f"R_ne_Vperp_{pulse}.txt", sep="\s+", names=["R (m)", "$n_{e}$ ($10^{19}m^{-3}$)", "$v_{\perp}$ (km/s)"])

    R=data['R (m)'].to_numpy()
    n=data['$n_{e}$ ($10^{19}m^{-3}$)'].to_numpy()
    v=data['$v_{\perp}$ (km/s)'].to_numpy()

    #order data in increasing R
    M=np.column_stack((R, v, n))
    M = M[M[:, 0].argsort()]

    R_s=M[:,0]
    v_s=M[:,1]
    n_s=M[:,2]
    
    return R_s,v_s,n_s

def crop_data(R, v, n, separatriz, to_left, to_right):

    new_R = R[ (R>=separatriz-to_left) & (R<=separatriz+to_right) ]
    new_v = v[ (R>=separatriz-to_left) & (R<=separatriz+to_right) ]
    new_n = n[ (R>=separatriz-to_left) & (R<=separatriz+to_right) ]

    return new_R, new_v, new_n

def SSE(x, y, fun):
    return np.sum((y-fun(x))**2)

def RSS(x, y, fun):
    return np.sqrt(SSE(x, y, fun))

def RMS(x, y, fun):
    return np.sqrt(SSE(x, y, fun)/len(x))


def find_spl_crit(spl, crit='max_absolute', interval=None):
    """Find splines maximum/minimum

    Args:
        spl (Univariate Spline object): _description_
        crit (str): 'max_absolute', 'min_absolute', 'max_rel' or 'min_rel'. '_rel' returns all critical values in the interval, 
                    '_absotule' returns the absolute critical value in interval. Defaults to 'max'.
        interval (array, optional): array with x_i and x_f over which to evaluete spline. Defaults to None.
    """
    assert len(interval)==2, 'interval must have len 2'
    assert interval[1]>interval[0], 'interval must be increasing'
    #compute spl's derivative
    spl_deriv=spl.derivative()
    
    #find crit values
    tck = spl._eval_args
    ppoly = PPoly.from_spline(tck)
    spl_deriv_roots=ppoly.derivative().roots()
    #print(spl_deriv_roots)
    
    #crop them to the desired interval
    spl_deriv_roots=spl_deriv_roots[(spl_deriv_roots>=interval[0]) & (spl_deriv_roots<=interval[1])]
    
    #evaluate second derivative
    second_deriv=spl_deriv.derivative()
    
    if crit=='max_absolute':
        maximums=spl_deriv_roots[second_deriv(spl_deriv_roots)<0]
        assert len(maximums)>0, 'No maximums were found'
        maximum=maximums[spl(maximums).argmax()]
        max_value=spl(maximum)
        return maximum, max_value
    
    if crit=='max_rel':
        maximums=spl_deriv_roots[second_deriv(spl_deriv_roots)<0]
        assert len(maximums)>0, 'No maximums were found'
        max_value=spl(maximums)
        #print(maximums)
        return maximums, max_value
    
    if crit=='min_absolute':
        minimums=spl_deriv_roots[second_deriv(spl_deriv_roots)>0]
        assert len(minimums)>0, 'No minimums were found'
        minimum=minimums[spl(minimums).argmin()]
        min_value=spl(minimum)
        return minimum, min_value
    
    if crit=='min_rel':
        minimums=spl_deriv_roots[second_deriv(spl_deriv_roots)>0]
        assert len(minimums)>0, 'No minimums were found'
        min_value=spl(minimums)
        return minimums, min_value


def find_spl_root(spl, root=0, interval=None):
    assert len(interval)==2, 'interval must have len 2'
    assert interval[1]>interval[0], 'interval must be increasing'
   
    tck = spl._eval_args
    ppoly = PPoly.from_spline(tck)
    roots = ppoly.solve(root)
    
    #crop them to the desired interval
    if interval is not None:
        roots=roots[(roots>=interval[0]) & (roots<=interval[1])]
    
    assert len(roots)>0, 'No roots were found'
    
    return roots




class fit_pulse:
    def __init__(self, pulse, df_pulse, 
                R=None, v=None, n=None,
                spl=None, n_knots=None, 
                spl_density=None, n_knots_density=6,
                spl_density_kwargs=dict(),
                spl_kwargs=dict(),
                get_data_kwargs=dict(), 
                R_multiplier=None,
                crop_data_kwargs=dict(to_left=0.04, to_right=0.02),
                shorten_data=True):
        
        
        self.pulse = pulse
        # Read data
        if (R is None) or (v is None) or (n is None):
            #print('Reading data from file')
            self.R, self.v, self.n = get_data(pulse, **get_data_kwargs)
        else:
            self.R = R
            self.v = v
            self.n = n

        
        # Define separatrix
        try:
            self.separatriz = df_pulse.loc[df_pulse['Pulse']==pulse, 'separatriz (m)'].values[0]
        except:
            self.separatriz = None
            print('No separatrix found')

        # Crop Data
        if shorten_data == True:
            self.R, self.v, self.n = crop_data(self.R, self.v, self.n, 
                                               self.separatriz, **crop_data_kwargs)

        if R_multiplier is not None:
            self.R = self.R * R_multiplier
            self.separatriz = self.separatriz * R_multiplier

        
        # Set Spline
        if spl is None:
            self.spl = constrained_spline(self.R , self.v , n_knots=n_knots, **spl_kwargs)
        else:
            self.spl=spl

        self.deriv = self.spl.derivative()
        self.knots = self.spl.get_knots()  
        self.n_knots = len(self.knots)

        try:
            self.bc_type = spl_kwargs['bc_type']
        except:
            self.bc_type = None

        # Set pulse information
        try:
            self.element = df_pulse.loc[df_pulse['Pulse']==pulse, 'Element'].values[0]
            self.power = df_pulse.loc[df_pulse['Pulse']==pulse, 'Power'].values[0]
        except:
            self.element = None
            self.power = None
            print('No element or power found')

        # Set Density Spline
        if spl_density is None:
            self.spl_density = constrained_spline(self.R, self.n, n_knots=n_knots_density,              
                                                    **spl_density_kwargs)
        else:
            self.spl_density = spl_density
        
        self.density_derivative = self.spl_density.derivative()
        self.density_2nd_derivative = self.spl_density.derivative(n=2)
        

        self.RSS = RSS(self.R, self.v, self.spl)
        self.RMS = np.sqrt(self.RSS / len(self.R))

    def curvature(self, x):
        return self.spl.derivative(n=2)(x) / (1+self.deriv(x)**2)**(3/2)
    
    def density_curvature(self, x):
        return self.spl_density.derivative(n=2)(x) / (1+self.density_derivative(x)**2)**(3/2)
    
    def deriv_integrate(self, x, ax=None, **kargs):
        """Returns the integral of derivative function between x[0] and x[1].
        If plots==True, then ax sould the the axis over which to color the integral.

        Args:
            x (array, len=2): interval over which to integrate
        """
        assert len(x)==2, 'the interval of integration must have length 2'

        integral = self.spl(x[1])-self.spl(x[0])
        
        if ax:
            x2=np.linspace(x[0], x[1], 1000)
            ax.fill_between(x2, 0, self.deriv(x2), zorder=0, **kargs)
            
        return integral

    def max_curv(self, num=1000, interval=None):
        """Returns curvature maximum and its location

        Args:
            unc (float, optional): step size. Defaults to 1e-5.

        Returns:
            x_max, curvature_max: position at which curvature is maximum and its value
        """
        if interval:
            x_eval=np.linspace(interval[0], interval[1], num)
        else:
            x_eval=np.linspace(self.R[0], self.separatriz, num)
            
        y_eval=self.curvature(x_eval)
        
        x_max=x_eval[np.argmax(y_eval)]
        
        return x_max, self.curvature(x_max)
    
    def max_shear(self, interval):
        arg_max, max = find_spl_crit(self.deriv, 'max_absolute', interval)
        return arg_max, max
    
    def min_curv(self, num=1000, interval=None):
        """Returns curvature minimum and its location

        Args:
            unc (float, optional): step size. Defaults to 1e-5.

        Returns:
            x_min, curvature_min: position at which curvature is minimum and its value
        """
        
        
        if interval:
            x_eval=np.linspace(interval[0], interval[1], num)
        else:
            maximum=self.max_curv(num)[0]
            x_eval=np.linspace(maximum, self.R[-1], num)
        
        y_eval=self.curvature(x_eval)
        
        x_min=x_eval[np.argmin(y_eval)]
        
        return x_min, self.curvature(x_min)

    def min_shear(self, interval):
        arg_min, min = find_spl_crit(self.deriv, 'min_absolute', interval)
        return arg_min, min

    def integrate_shear_by_curvature_same_distance(self, ax_shear=None, ax_curv=None, average=False):

        """integrate the fits velocity taking as integration limits the maximum and minimum of the fits curvature.
        inner shear is defined as the integral taken over that distance to the left of the maximum
        outter shear is taken as the integral between maximum and minimum
        scrape off is taken as the integral taken over that distance to the left of the maximum 

        Args:
            plots (bool, optional): use if wish to plot que integral. Defaults to False.
            ax (_type_, optional): specify the axis for shear over which to plot the integral. Defaults to None.

        Returns:
            float, float, float: inner_shear, outter_shear, scrape_off
        """
        #global minimum 
        x_c_min = self.min_curv()[0]
            
        #maximum to the left of minimum
        x_c_max = self.max_curv()[0]

        distance=x_c_min-x_c_max

        if average==False:
            inner_shear=self.deriv_integrate(x=[x_c_max-distance, x_c_max], ax=ax_shear)
            outter_shear=self.deriv_integrate(x=[x_c_max, x_c_min], ax=ax_shear)
            scrape_off=self.deriv_integrate(x=[x_c_min, x_c_min+distance], ax=ax_shear)
        else:
            inner_shear=self.deriv_integrate(x=[x_c_max-distance, x_c_max], ax=ax_shear)/distance
            outter_shear=self.deriv_integrate(x=[x_c_max, x_c_min], ax=ax_shear)/distance
            scrape_off=self.deriv_integrate(x=[x_c_min, x_c_min+distance], ax=ax_shear)/distance
        
        if ax_curv:
            ax_curv.scatter(np.array([x_c_min,x_c_max]), self.curvature(np.array([x_c_min,x_c_max])))
            
        #print(integral1,integral2,integral3)
            
        return inner_shear, outter_shear, scrape_off

    def integrate_shear_by_velocity_same_distance(self, ax_shear=None):
        """integrate the fits velocity taking as integration limits the maximum and minimum of the fits velocity.
        inner shear is defined as the integral taken over that distance to the left of the maximum
        outter shear is taken as the integral between maximum and minimum
        scrape off is taken as the integral taken over that distance to the left of the maximum 

        Args:
            plots (bool, optional): use if wish to plot que integral. Defaults to False.
            ax (_type_, optional): specify the axis for shear over which to plot the integral. Defaults to None.

        Returns:
            float, float, float: inner_shear, outter_shear, scrape_off
        """
        #global minimum 
        x_v_min = find_spl_crit(self.spl, interval=[self.R[0], self.R[-1]], crit='min_absolute')[0]
            
        #maximum to the left of minimum
        x_v_max = find_spl_crit(self.spl, interval=[x_v_min, self.R[-1]], crit='max_rel')[0][0]

        distance=x_v_max-x_v_min

        inner_shear=self.deriv_integrate([x_v_min-distance, x_v_min], ax=ax_shear)
        outter_shear=self.deriv_integrate([x_v_min, x_v_max], ax=ax_shear)
        scrape_off=self.deriv_integrate([x_v_max, x_v_max+distance], ax=ax_shear)
            
        #print(integral1,integral2,integral3)
            
        return inner_shear, outter_shear, scrape_off
    
    def integrate_velocity_and_curvature(self, plots=False, ax_shear=None):
        
        #global minimum 
        x_v_min = find_spl_crit(self.spl, interval=[self.R[0], self.R[-1]], crit='min_absolute')[0]
            
        #maximum to the left of minimum
        x_v_max = find_spl_crit(self.spl, interval=[x_v_min, self.R[-1]], crit='max_rel')[0][0]
        
        #maximum curvature to the right of minimum
        x_c_max = self.max_curv(interval=[x_v_max, self.R[-1]])[0]
        
        #minimum curvature to the left of minimum
        x_c_min = self.min_curv(interval=[self.R[0], x_v_min])[0]
        
        inner_shear=self.deriv_integrate([x_c_min, x_v_min], ax=ax_shear)
        outter_shear=self.deriv_integrate([x_v_min, x_v_max], ax=ax_shear)
        scrape_off=self.deriv_integrate([x_v_max, x_c_max], ax=ax_shear)
            
        #print(integral1,integral2,integral3)
            
        return inner_shear, outter_shear, scrape_off

    def integrate_shear_by_curvature(self, ax_shear=None, ax_curv=None, average=False):
        
        curv_max=self.max_curv()[0]
            
        curv_min=self.min_curv()[0]
            
        curv_min_2=self.min_curv(interval=[self.R[0], curv_max])[0]
            
        curv_max_2=self.max_curv(interval=[curv_min, self.R[-1]])[0]
        
        if average==False: 
            inner=self.deriv_integrate(x=[curv_min_2, curv_max], ax=ax_shear)
            outter=self.deriv_integrate(x=[curv_max, curv_min], ax=ax_shear)
            scrape=self.deriv_integrate(x=[curv_min, curv_max_2], ax=ax_shear)
        else:
            inner=self.deriv_integrate(x=[curv_min_2, curv_max], ax=ax_shear) / (curv_max - curv_min_2)
            outter=self.deriv_integrate(x=[curv_max, curv_min], ax=ax_shear) / (curv_min - curv_max)
            scrape=self.deriv_integrate(x=[curv_min, curv_max_2], ax=ax_shear) / (curv_max_2 - curv_min)
            
        if ax_curv:
            ax_curv.scatter(np.array([curv_min_2, curv_max, curv_min, curv_max_2]), self.curvature(np.array([curv_min_2, curv_max, curv_min, curv_max_2])))
            
        #print(integral1,integral2,integral3)
            
        return inner, outter, scrape
    
    def max_shear_between_curvature_extremes(self):
        
        curv_max=self.max_curv()[0]
            
        curv_min=self.min_curv()[0]
            
        curv_min_2=self.min_curv(interval=[self.R[0], curv_max])[0]
            
        curv_max_2=self.max_curv(interval=[curv_min, self.R[-1]])[0]
            
        inner = self.min_shear([curv_min_2, curv_max])[1]
        outter=self.max_shear([curv_max, curv_min])[1]
        scrape=self.min_shear([curv_min, curv_max_2])[1]
        
        return inner, outter, scrape
    
    def max_shear_same_distance_curvature(self):
        #global minimum 
        x_c_min = self.min_curv()[0]
            
        #maximum to the left of minimum
        x_c_max = self.max_curv()[0]

        distance=x_c_min-x_c_max

        inner_max=self.max_shear([x_c_max-distance, x_c_max])[1]
        outter_max=self.max_shear([x_c_max, x_c_min])[1]
        scrap_max=self.max_shear([x_c_min, x_c_min+distance])[1]
                    
        return inner_max, outter_max, scrap_max
    
    def max_shear_density_curvature(self, tol=0.1):
        #global minimum 
        x_c_min = self.min_curv()[0]
            
        #maximum to the left of minimum
        x_c_max = self.max_curv()[0]
        
        density_curvature_root1 = root_scalar(lambda x: np.abs(self.density_curvature(x)) - tol, x0 = self.R[0], method='newton').root
        density_curvature_root2 = root_scalar(lambda x: np.abs(self.density_curvature(x)) - tol, x0 = self.R[-1], method='newton').root

        inner_max=self.min_shear([density_curvature_root1, x_c_max])[1]
        outter_max=self.max_shear([x_c_max, x_c_min])[1]
        scrap_max=self.min_shear([x_c_min, x_c_min+density_curvature_root2])[1]
                    
        return inner_max, outter_max, scrap_max
    
    def integrate_shear_by_density_curvature(self, tol=0.05, ax_shear=None, ax_density_curv=None, average=False):
        
        #density curvature roots approximately
        density_curvature_root1 = root_scalar(lambda x: (self.density_curvature(x)) + tol, x0 = self.R[0], method='newton').root
        density_curvature_root2 = root_scalar(lambda x: (self.density_curvature(x)) - tol, x0 = self.R[-1], method='newton').root
        
        #maximums and minimums of curvature
        curv_max=self.max_curv()[0] 
        curv_min=self.min_curv()[0]
        
        if average==False:
            inner=self.deriv_integrate(x=[density_curvature_root1, curv_max], ax=ax_shear)
            outter=self.deriv_integrate(x=[curv_max, curv_min], ax=ax_shear)
            scrape=self.deriv_integrate(x=[curv_min, density_curvature_root2], ax=ax_shear)
        else:
            inner=self.deriv_integrate(x=[density_curvature_root1, curv_max], ax=ax_shear)/(curv_max - density_curvature_root1)
            outter=self.deriv_integrate(x=[curv_max, curv_min], ax=ax_shear)/(curv_min - curv_max)
            scrape=self.deriv_integrate(x=[curv_min, density_curvature_root2], ax=ax_shear)/(density_curvature_root2 - curv_min)
        
        if ax_density_curv:
            ax_density_curv.scatter(np.array([density_curvature_root1, density_curvature_root2, curv_max, curv_min]), self.density_curvature(np.array([density_curvature_root1, density_curvature_root2, curv_max, curv_min])))
        
        return inner, outter, scrape
   
    def integrate_shear_by_density_2nd_derivative(self, tol=0.05, ax_shear=None, ax_density_2nd_derivative=None, average=False):
        #density curvature roots approximately
        density_2nd_derivative_root1 = find_spl_root(self.density_2nd_derivative, root=-tol, interval=[self.R[0], self.R[-1]])[0]
        density_2nd_derivative_root2 = find_spl_root(self.density_2nd_derivative, root=tol, interval=[self.R[0], self.R[-1]])[-1]
        
        #maximums and minimums of 2nd_derivative
        curv_max=self.max_curv()[0] 
        curv_min=self.min_curv()[0]
        
        if average==False:
            inner=self.deriv_integrate(x=[density_2nd_derivative_root1, curv_max], ax=ax_shear)
            outter=self.deriv_integrate(x=[curv_max, curv_min], ax=ax_shear)
            scrape=self.deriv_integrate(x=[curv_min, density_2nd_derivative_root2], ax=ax_shear)
        else:
            inner=self.deriv_integrate(x=[density_2nd_derivative_root1, curv_max], ax=ax_shear)/(curv_max - density_2nd_derivative_root1)
            outter=self.deriv_integrate(x=[curv_max, curv_min], ax=ax_shear)/(curv_min - curv_max)
            scrape=self.deriv_integrate(x=[curv_min, density_2nd_derivative_root2], ax=ax_shear)/(density_2nd_derivative_root2 - curv_min)
        
        if ax_density_2nd_derivative:
            ax_density_2nd_derivative.scatter(np.array([density_2nd_derivative_root1, density_2nd_derivative_root2, curv_max, curv_min]),\
                self.density_2nd_derivative(np.array([density_2nd_derivative_root1, density_2nd_derivative_root2, curv_max, curv_min])))
        
        return inner, outter, scrape