from astropy.io import fits
import glob
import numpy as np
import matplotlib.pyplot as plt
import regions as rg

class FitsImage:
    
    def __init__(self, fn, rf, instrument='xmm', output=None, frame=0):
        self.file_name = fn
        self.frame = frame
        
        ds9_regions = rg.Regions.read(rf, format='ds9')
        self.annulus = ds9_regions.regions[0]
        self.mask_annulus = self.annulus.to_mask()      
        self.inner_circle = rg.CirclePixelRegion(self.annulus.center, self.annulus.inner_radius)
        self.mask_inner = self.inner_circle.to_mask()
        
        self.mean = None
        self.sigma = None
             
        if output is None:
            self.output = fn+'.lev' 
        else:
            self.output = output
            
        # it would be good to read out the instrument name from fits header.
        with fits.open(self.file_name) as hdu:
            if (instrument=='tgss') or (instrument=='nvss'):
                self.img = hdu[0].data[0][0]
            elif instrument=='xmm':
                # it would be a good idea to recognize colim (hdu[0].data[0]) or oimage (hdu[0].data) (Oimage has wrong wcs).
                # Can use self.frame==None for that.
                img = hdu[0].data[self.frame]
                self.img = np.nan_to_num(img)
            else:
                raise ValueError('Instrument not identified.')
        
        
    def define_sigma_max(self, sigma_max):
        if sigma_max=='inner':
            img_inner = self.mask_inner.multiply(self.img)
            new_sigma_max = (np.amax(img_inner)-self.mean)/self.sigma
            print('inner radius being used')
            print(new_sigma_max)
        elif sigma_max=='full':
            new_sigma_max = (np.amax(self.img)-self.mean)/self.sigma
            print('full image being used')
            print(new_sigma_max)
        elif isinstance(sigma_max, int):
            new_sigma_max = sigma_max
            print('Using user-defined sigma_max={}'.format(new_sigma_max))
        else:
            raise ValueError('Invalid option.')
        return new_sigma_max
                  
        
    def calculate_levels(self, sigma0=1, sigma_max=5, n_levels=None, sigma_step=1):
        # implement log contours.
        img_masked = self.mask_annulus.multiply(self.img)
        self.mean = img_masked.mean()
        self.sigma = img_masked.std()
        sigma_max = self.define_sigma_max(sigma_max)  
        if n_levels:
            print('Using linspace')
            levels = self.mean + self.sigma * np.linspace(sigma0,sigma_max,n_levels)
            # levels = [i*self.sigma + self.mean for i in np.linspace(sigma0,sigma_max,n_levels)]
        else:
            print('Using arange')
            levels =  self.mean + self.sigma * np.arange(sigma0,sigma_max,sigma_step)
        print(levels)
        np.savetxt(self.output, levels)
        
                
    ###define plotting functions

    def plot_annulus(self):
        fig, ax = plt.subplots()
        img_masked = self.mask_inner.multiply(self.img)
        ind_max, val_max = np.unravel_index(np.argmax(img_masked, axis=None), img_masked.shape), np.max(img_masked)
        print(ind_max, val_max)
        ind_max_img = (self.inner_circle.bounding_box.ixmin+ind_max[1], self.inner_circle.bounding_box.iymin+ind_max[1])
        cir = plt.Circle(ind_max_img, 4, color='r',fill=True)
        ax.add_patch(cir)
        ax.imshow(self.img, vmin=0, vmax=4*self.sigma, cmap='Greys', origin='lower')
        # ax.imshow(self.img, vmin=0, vmax=self.mean+self.define_sigma_max('inner')*self.sigma, 
        #           cmap='Greys', origin='lower')
        self.annulus.plot(ax=ax, edgecolor='blue', fill=False, lw=1)
        plt.show()
        
    def plot_region(self, region):
        if region=='bkg':
            img_masked = self.mask_annulus.multiply(self.img)
        elif region=='inner':
            img_masked = self.mask_inner.multiply(self.img)
        else:
            raise ValueError('Invalid option.')
        fig, ax = plt.subplots()
        ax.imshow(img_masked, vmin=0, vmax=self.mean+4*self.sigma, 
                  cmap='Greys', origin='lower')
        plt.show()
        