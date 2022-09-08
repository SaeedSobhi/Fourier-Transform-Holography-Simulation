import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
import skimage.transform #to install it: pip install scikit-image
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from ipywidgets import interact

#DeteactorShape = 1: Circle, 2: Square
#The bigger the PinholeFactor the sammler the pinhole
#Ydistance and Xdistance incate the propagation distance to the pinhole
#N is the quantaization level
#dx: detector side length factor, we multiply it with N

im = imread("Cameraman.jpg", pilmode="L") #jpg image should be in the same folder or you have to include the whole directory
im = skimage.transform.resize(im, [256, 256], anti_aliasing=False)
im.astype(float)

def SimulateFTH(im, DetectorShape=1, PinholeFactor=200, Ydistance=85, Xdistance=85, N=2**8, dx=1, Brightness=0):
    xgrid = []
    ygrid = []
    for i in range(-N // 2, N // 2):  # use '//' for an int result, '/' for float result
        xgrid.append(i)
        ygrid.append(i)
    L = N * dx  # side length
    [X, Y] = np.meshgrid(xgrid, ygrid)  # 2D coordinates
    # R = np.sqrt(X ** 2 + Y ** 2)
    # theta = np.arctan2(Y, X)
    im = skimage.transform.resize(im, [256, 256], anti_aliasing=False)
    im.astype(float)
    im = im + Brightness

    n = 8
    if DetectorShape == 1:
        E = abs(X) ** 2 + abs(Y) ** 2 < 1.5 * ((1 / n * L) ** 2)
    else:
        E = (abs(X) < 1 / n * L) * (abs(Y) < 1 / n * L)

    #pinhole
    r = np.sqrt(X ** 2 + Y ** 2) < L / PinholeFactor

    # propagation distance
    r = np.roll(r, Xdistance, 1)
    r = np.roll(r, -Ydistance, 0)

    #the detected image through the pinhole
    E1 = E * im + r

    #detector intensity
    H = np.fft.fftshift(abs(np.fft.fft2(E1)) ** 2)

    # normalize the diffraction pattern and add poisson noise
    H = H / np.max(H)  # normalizing [0,1]
    bit_depth = 16
    H = 2 ** bit_depth * H  # normalize to bit range of detector, new range [0, 2^bit_depth]
    H_noisy = np.random.poisson(H).astype(float)

    # autocorrelation
    acf = np.fft.fftshift(np.fft.ifft2(H_noisy))  # H'_noisy
    lgacf = np.log10(abs(acf) + 1e-3)

    # back propagating
    # acf1 = np.fft.fft2(acf)
    # acf1 = np.fft.fftshift(acf1)
    acf1 = np.fft.fftshift(np.fft.ifft2(H))
    # acf1[acf1 > 0.9] = 0 #filter option

    lgacf1 = np.log10(abs(acf1) + 1e-3)

    # im = skimage.transform.resize(im, [256, 256], anti_aliasing=False)
    # E2 = E * im + r

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))

    plt.subplots_adjust(left=0.25, bottom=0.3)
    textstr = 'a: Original image,\n(the object)\nb: the illuminated area of\nthe object \nplus the pinhole \nc:normalized ' \
              'diffraction pattern \n(noise free) \n' \
              'd: H_noisy \nH after adding\npoissonian noise to it \ne: H\'_noisy: IFT H_noisy \nf:H\': IFT of H'
    plt.gcf().text(0.02, 0.5, textstr, fontsize=14)

    #plt.figure(1)
    axs[0, 0].imshow(im, cmap=plt.cm.gray, interpolation='nearest')
    axs[0, 0].set_title("a")

    axs[0, 1].imshow(E1, cmap=plt.cm.gray, interpolation='nearest')
    axs[0, 1].set_title("b")

    axs[0, 2].imshow(np.log10(H + 1e-3), cmap=plt.cm.gray, interpolation='nearest')
    axs[0, 2].set_title("c")

    axs[1, 0].imshow(np.log10(H_noisy + 1e-3), cmap=plt.cm.gray, interpolation='nearest')
    axs[1, 0].set_title("d")

    axs[1, 1].imshow(lgacf, cmap=plt.cm.gray, interpolation='nearest')
    axs[1, 1].set_title("e")

    axs[1, 2].imshow(lgacf1, cmap=plt.cm.gray, interpolation='nearest')
    axs[1, 2].set_title("f")

    E_shifted2 = np.roll(E,  -Xdistance, 1)
    E_shifted2 = np.roll(E_shifted2, Ydistance, 0)

    Shifted_reconstructed_image = E_shifted2 * lgacf
    reconstructed_image = np.roll(Shifted_reconstructed_image, Xdistance, 1)
    reconstructed_image = np.roll(reconstructed_image, -Ydistance, 0)

    Shifted_reconstructed_noisefree_image = E_shifted2 * lgacf1
    reconstructed_noisefree_image = np.roll(Shifted_reconstructed_noisefree_image, Xdistance, 1)
    reconstructed_noisefree_image = np.roll(reconstructed_noisefree_image, -Ydistance, 0)

    OI = np.log10(E * im + 1e-3)
    OI[OI < -2] = 0
    # fig2, axs2 = plt.subplots(1, 3, figsize=(16, 9))
    # plt.figure(2)
    #
    # axs2[0].imshow(reconstructed_image, cmap=plt.cm.gray, interpolation='nearest')
    # axs2[0].set_title("reconstructed_noisy_image")
    #
    # axs2[1].imshow(reconstructed_noisefree_image, cmap=plt.cm.gray, interpolation='nearest')
    # axs2[1].set_title("reconstructed_image_noise-free")
    #
    # axs2[2].imshow(OI, cmap=plt.cm.gray, interpolation='nearest')
    # axs2[2].set_title("original image on the detector")


    YS = fig.add_axes([0.25, 0.09, 0.65, 0.03])
    XS = fig.add_axes([0.25, 0.12, 0.65, 0.03])
    PS = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    #BS = fig.add_axes([0.25, 0.18, 0.65, 0.03])
    DS = fig.add_axes([0.25, 0.18, 0.07, 0.03])

    YdistanceSlider = Slider(YS, 'Ydistance', -100, 100, valstep=1, valinit=Ydistance)
    XdistanceSlider = Slider(XS, 'Xdistance', -100, 100, valstep=1, valinit=Xdistance)
    PinholeSLider = Slider(PS, 'PinholeFactor', 1, 1000, valstep=10, valinit=PinholeFactor)
    DetectorSlider = Slider(DS, 'Detector-Shape', 1, 2, valstep=1, valinit=DetectorShape)
    #BrightnessSlider = Slider(BS, 'Adjust Brightness', -10, 10, valstep=1, valinit=Brightness)


    # DetectorLabels = ('Circular', 'Square')
    # RadioButtons(DS, DetectorLabels, [1, 2])


    def update(val):
        Xdistance = XdistanceSlider.val
        Ydistance = YdistanceSlider.val
        PinholeFactor = PinholeSLider.val
        DetectorShape = DetectorSlider.val
        #Brightness = BrightnessSlider.val


        if DetectorShape == 1:
            E = abs(X) ** 2 + abs(Y) ** 2 < 1.5 * ((1 / n * L) ** 2)
        else:
            E = (abs(X) < 1 / n * L) * (abs(Y) < 1 / n * L)

        # pinhole
        r = np.sqrt(X ** 2 + Y ** 2) < L / PinholeFactor

        # propagation distance
        r = np.roll(r, Xdistance, 1)
        r = np.roll(r, -Ydistance, 0)

        im1 = im + Brightness

        # the detected image through the pinhole
        E1 = E * im1 + r

        # detector intensity
        H = np.fft.fftshift(abs(np.fft.fft2(E1)) ** 2)

        # normalize the diffraction pattern and add poisson noise
        H = H / np.max(H)  # normalizing [0,1]
        bit_depth = 16
        H = 2 ** bit_depth * H  # normalize to bit range of detector, new range [0, 2^bit_depth]
        H_noisy = np.random.poisson(H).astype(float)

        # autocorrelation
        acf = np.fft.fftshift(np.fft.ifft2(H_noisy))  # H'_noisy
        lgacf = np.log10(abs(acf) + 1e-3)

        # back propagating
        # acf1 = np.fft.fft2(acf)
        # acf1 = np.fft.fftshift(acf1)
        acf1 = np.fft.fftshift(np.fft.ifft2(H))
        # acf1[acf1 > 0.9] = 0 #filter option

        lgacf1 = np.log10(abs(acf1) + 1e-3)
        axs[0, 0].imshow(im, cmap=plt.cm.gray, interpolation='nearest')
        axs[0, 0].set_title("a")

        axs[0, 1].imshow(E1, cmap=plt.cm.gray, interpolation='nearest')
        axs[0, 1].set_title("b")

        axs[0, 2].imshow(np.log10(H + 1e-3), cmap=plt.cm.gray, interpolation='nearest')
        axs[0, 2].set_title("c")

        axs[1, 0].imshow(np.log10(H_noisy + 1e-3), cmap=plt.cm.gray, interpolation='nearest')
        axs[1, 0].set_title("d")

        axs[1, 1].imshow(lgacf, cmap=plt.cm.gray, interpolation='nearest')
        axs[1, 1].set_title("e")

        axs[1, 2].imshow(lgacf1, cmap=plt.cm.gray, interpolation='nearest')
        axs[1, 2].set_title("f")

        E_shifted2 = np.roll(E, -Xdistance, 1)
        E_shifted2 = np.roll(E_shifted2, Ydistance, 0)

        Shifted_reconstructed_image = E_shifted2 * lgacf
        reconstructed_image = np.roll(Shifted_reconstructed_image, Xdistance, 1)
        reconstructed_image = np.roll(reconstructed_image, -Ydistance, 0)

        Shifted_reconstructed_noisefree_image = E_shifted2 * lgacf1
        reconstructed_noisefree_image = np.roll(Shifted_reconstructed_noisefree_image, Xdistance, 1)
        reconstructed_noisefree_image = np.roll(reconstructed_noisefree_image, -Ydistance, 0)

        OI = np.log10(E * im + 1e-3)
        OI[OI < -2] = 0
        # # fig2, axs2 = plt.subplots(1, 3, figsize=(16, 9))
        # # plt.figure(2)
        #
        # axs2[0].imshow(reconstructed_image, cmap=plt.cm.gray, interpolation='nearest')
        # axs2[0].set_title("reconstructed_noisy_image")
        #
        # axs2[1].imshow(reconstructed_noisefree_image, cmap=plt.cm.gray, interpolation='nearest')
        # axs2[1].set_title("reconstructed_image_noise-free")
        #
        # axs2[2].imshow(OI, cmap=plt.cm.gray, interpolation='nearest')
        # axs2[2].set_title("original image on the detector")

    YdistanceSlider.on_changed(update)
    XdistanceSlider.on_changed(update)
    PinholeSLider.on_changed(update)
    DetectorSlider.on_changed(update)
    #BrightnessSlider.on_changed(update)
    plt.show()


'''
#autocorrelation
N = 2 ** 8 #quantization level
dx = 1
xgrid = []
ygrid = []
for i in range(-N//2, N//2):   #use '//' for an int result, '/' for float result
    xgrid.append(i)
    ygrid.append(i)
L = N * dx  # side length
[X, Y] = np.meshgrid(xgrid, ygrid)  #2D coordinates
R = np.sqrt(X ** 2 + Y ** 2)
theta = np.arctan2(Y, X)
im = np.cos(23*theta) > 0


'''

''' Square detector 
n = 8
E = (abs(X) < 1/n*L) * (abs(Y) < 1/n*L)

r = np.sqrt(X ** 2 + Y ** 2) < L/200
r2 = np.sqrt(X ** 2 + Y ** 2) < L/200

r = np.roll(r, 100)
E1 = E * im  + r
#E2 = E * im + r
fig, axs = plt.subplots(2, 4, figsize=(16, 9))
plt.subplots_adjust(left=0.4, bottom=0.4)

I = np.fft.fftshift(abs(np.fft.fft2(E1))**2) #detector intensity

# add poisson noise
I = I / np.max(I) #normalize to range [0,1]
bit_depth = 16
I = 2**bit_depth * I # normalize to bit range of detector, new range [0, 2^bit_depth]
I_noisy = np.random.poisson(I).astype(float)

#autocorrelation
acf = np.fft.fftshift(np.fft.ifft2(I_noisy))
lgacf = np.log10(abs(acf) + 1e-3)

acf1 = np.fft.fftshift(np.fft.ifft2(I))
lgacf1 = np.log10(abs(acf) + 1e-3)

axs[0, 0].imshow(im, cmap=plt.cm.gray, interpolation='nearest')
axs[0, 0].set_title("Original image")

axs[0, 1].imshow(E*(1-r2), cmap=plt.cm.gray, interpolation='nearest')
axs[0, 1].set_title("The detector")

axs[0, 2].imshow(1-r2, cmap=plt.cm.gray, interpolation='nearest')
axs[0, 2].set_title("1-r2") # what is the name of this

axs[0, 3].imshow(r, cmap=plt.cm.gray, interpolation='nearest')
axs[0, 3].set_title("The Pinhole")

axs[0, 3].imshow(E1, cmap=plt.cm.gray, interpolation='nearest')
axs[0, 3].set_title("E1 = detector * im * (1-r2) + pinhole")

axs[1, 0].imshow(np.log10(I + 1e-3), cmap=plt.cm.gray, interpolation='nearest')
axs[1, 0].set_title("I = FT of E1 after shifting")

axs[1, 1].imshow(np.log10(I_noisy + 1e-3), cmap=plt.cm.gray, interpolation='nearest')
axs[1, 1].set_title("I_noisy")

axs[1, 2].imshow(lgacf, cmap=plt.cm.gray, interpolation='nearest')
axs[1, 2].set_title("IFT of I_noisy")

axs[1, 3].imshow(lgacf1, cmap=plt.cm.gray, interpolation='nearest')
axs[1, 3].set_title("IFT of I")

plt.show()
'''

'''    
n = 8
#E = (abs(X) < 1.5/n*L) * (abs(Y) < 1.5/n*L) #square
E =  abs(X)**2 + abs(Y)**2 < 1.5*((1/n*L )**2)
#Frame = np.zeros([1024,1024])
r = np.sqrt(X ** 2 + Y ** 2) < L/200 # Pinhole
#r2 = np.sqrt(X ** 2 + Y ** 2) < L/200

#propagation distance
Ydistance = 85
Xdistance = 85


r = np.roll(r, Xdistance, 1)
r = np.roll(r, -Ydistance, 0)
E1 =  E * im  + r


#E1 = E * (np.abs(np.fft.fft2(im)  + np.fft.fft2(r)))**2



#I = np.fft.fftshift(E1) #detector intensity
H = np.fft.fftshift(abs(np.fft.fft2(E1))**2) #detector intensity

# normalize the diffraction pattern and add poisson noise
H = H / np.max(H) #normalizing [0,1]
bit_depth = 16
H = 2**bit_depth * H # normalize to bit range of detector, new range [0, 2^bit_depth]
H_noisy = np.random.poisson(H).astype(float)

#filtering


#autocorrelation
acf = np.fft.fftshift(np.fft.ifft2(H_noisy)) #H'_noisy
lgacf = np.log10(abs(acf) + 1e-3)

#back propagating
acf1 = np.fft.fft2(acf)
acf1 = np.fft.fftshift(acf1)


acf1 = np.fft.fftshift(np.fft.ifft2(H))
#acf1[acf1 > 0.9] = 0

lgacf1 = np.log10(abs(acf) + 1e-3)

#im = skimage.transform.resize(im, [256, 256], anti_aliasing=False)
#E2 = E * im + r

fig, axs = plt.subplots(2, 3, figsize=(16, 9))
plt.subplots_adjust(bottom=0.3)


axs[0, 0].imshow(im, cmap=plt.cm.gray, interpolation='nearest')
axs[0, 0].set_title("Original image, the object")

axs[0, 1].imshow(E1, cmap=plt.cm.gray, interpolation='nearest')
axs[0, 1].set_title("The object on the detector and the pinhole")

axs[0, 2].imshow(np.log10(H + 1e-3), cmap=plt.cm.gray, interpolation='nearest')
axs[0, 2].set_title("H(q)")

axs[1, 0].imshow(np.log10(H_noisy + 1e-3), cmap=plt.cm.gray, interpolation='nearest')
axs[1, 0].set_title("H_noisy: H after adding poissonian noise to it")

axs[1, 1].imshow(lgacf, cmap=plt.cm.gray, interpolation='nearest')
axs[1, 1].set_title("H'_noisy: IFT H_noisy")

axs[1, 2].imshow(lgacf1, cmap=plt.cm.gray, interpolation='nearest')
axs[1, 2].set_title("H': IFT of H")
'''



#SimulateFTH(im=im, axs01=axs01, axs02=axs02, DeteactorShape=1, PinholeFactor=200, Ydistance=Ydistance1, Xdistance=Xdistance1, N=2**8, dx=1)

# interact(SimulateFTH, im=im, DeteactorShape=(1, 2), PinholeFactor=200,
#          Ydistance=(-100, 100, 1), Xdistance=(-100, 100, 1), N=2**8, dx=1)

SimulateFTH(im = im)
'''
YdistanceSlider = Slider(ax=,valmin=-100, valmax=100, valinit=Ydistance1, valstep=1)
XdistanceSlider = Slider(valmin=-100, valmax=100, valinit=Xdistance1, valstep=1)

def update(val):
    Ydistance1 = YdistanceSlider.val
    #freq = sfreq.val
    #l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    #r = np.roll(r, -Ydistance, 0)
    SImulateFTH(im=im, axs1=axs01, axs2=axs02, DeteactorShape=1, PinholeFactor=200,
                Ydistance=Ydistance1, Xdistance=Xdistance1, N=2 ** 8, dx=1)
    fig.canvas.draw_idle()
'''

# sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0, valstep=delta_f)
# samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)
#
# def update(val):
#     amp = samp.val
#     freq = sfreq.val
#     l.set_ydata(amp*np.sin(2*np.pi*freq*t))
#     fig.canvas.draw_idle()
#
# #sliders:
# #position of the pinhole  two sliders, vertical (0 to y.length) and horizental (0 to x.length) distance
# #shape of the detector, choices
# #size of the detector, changes automatically with the propagation distance
# #pinhole size
#
# sfreq.on_changed(update)
# samp.on_changed(update)


#plt.show()



