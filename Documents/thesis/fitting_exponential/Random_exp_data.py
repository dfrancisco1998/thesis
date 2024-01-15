import numpy as np
import random as rdm

def genereate_T1_data(n_samples = 101, noise_lvl = 0.03 , max_stretch = 0.2,lin_log = 0):
    #linspace or logspace between 0 and 1
    if lin_log:
        x_data = (np.linspace(0,1,n_samples))
    else:
        x_data = (np.logspace(-1,0,n_samples))
        x_data = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

    # generate random T1 value
    T1 = rdm.random()*0.5

    # add some stretching
    T1_stretch = 1+(rdm.random()-0.5)*max_stretch

    # create decay and add noise
    T1_data = np.exp(-(x_data/T1)**T1_stretch) +np.random.normal(0,noise_lvl,n_samples)

    return x_data, T1_data


def genereate_T2_data(n_samples = 101, noise_lvl = 0.03 ,max_stretch = 0.2, lin_log = 1):
    #linspace or logspace between 0 and 1
    if lin_log:
        x_data = (np.linspace(0,1,n_samples))
    else:
        x_data = (np.logspace(-1,0,n_samples))
        x_data = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

    # generate random T1 value
    T2 = rdm.random()*0.5

    # add some stretching
    T2_stretch = 1+(rdm.random()-0.5)*max_stretch

    # create decay 
    T2_data = np.exp(-(x_data/T2)**T2_stretch) 
    
    # add 13C bumps
    bump_spacing = rdm.random()*0.15+0.15
    bump_amp = rdm.random()*0.4+0.6
    T2_data = np.multiply( np.exp(-(x_data/T2)**T2_stretch), (np.cos(2*np.pi/bump_spacing*x_data)/2+0.5)*bump_amp+(1-bump_amp) )

    # add noise
    T2_data = T2_data + np.random.normal(0,noise_lvl,n_samples)

    return x_data, T2_data


def genereate_T2star_data(n_samples = 101, noise_lvl = 0.03 , max_stretch = 0.2,lin_log = 1):
    #linspace or logspace between 0 and 1
    if lin_log:
        x_data = (np.linspace(0,1,n_samples))
    else:
        x_data = (np.logspace(-1,0,n_samples))
        x_data = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

    # generate random T1 value
    T2 = rdm.random()*0.5

    # add some stretching
    T2_stretch = 1+(rdm.random()-0.5)*max_stretch

    # create decay 
    T2_data = np.exp(-(x_data/T2)**T2_stretch) 
    
    # add 13C bumps
    bump_spacing = rdm.random()*0.1+0.1
    T2_data = np.multiply( np.exp(-(x_data/T2)**T2_stretch), np.cos(2*np.pi/bump_spacing*x_data) )

    # add noise
    T2_data = T2_data + np.random.normal(0,noise_lvl,n_samples)

    return x_data, T2_data

if __name__ == '__main__':
    import matplotlib.pyplot as mtp
    [x, data] = genereate_T2star_data()
    mtp.plot(x,data)
    mtp.show()