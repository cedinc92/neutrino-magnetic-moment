import numpy as np
from scipy.stats import gaussian_kde
from scipy import integrate
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import math
from bisect import bisect_left

output_filename = "electron_energy_angle.tex"
class NuMaSClass:
    def __init__(self, eventNum,numag):
        self.kde = None
        self.x_values = None
        self.cdf = None
        #self.inverse_cdf = None  # Initialize inverse_cdf
        self.interpolator = None
        # Physical constants (Natural Units: hbar = c = 1)
        self.GF= 1.166e-5  # GeV^(-2) (Fermi constant)
        self.sin2_theta_w = 0.231  # Weak mixing angle (sin^2(theta_W))
        self.fine_struct=0.00729927  # fine structure constant
        self.eventNum=eventNum
        self.p3=[]                         #will later contain the outgoing momenta
        self.p4=[]
        self.weights=[]
        self.me=0.511e-3
        self.moment= numag
        
    def clean(self):                       #used to reset the class variables for multiple runs of the generator
        self.p3=[]
        self.p4=[]
        self.weights=[]
        
    def addition(self, a, b):       #defines 4momentum addition
        return [a[0]+b[0],a[1]+b[1],a[2]+b[2],a[3]+b[3]]
        
    def scalarProduct(self, a, b):  #defines 4momentum scalar product
        return a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3]   
       # Define the integrand function

    def averaged_diff_cross_section(self, E_nu, T):
        # Numerator: Integral of dσ/dT weighted by the neutrino spectrum
        crossR=[0,0,0]
        E_min=0.1
        E_max=2000
        mpv=250

        term = self.diff_cross_section(E_nu, T)  
 
        def integrand0(E_nu):
            return term[3] * self.neutrino_spectrumfnc(E_nu, mpv)

        def integrand1(E_nu):
            return term[4] * self.neutrino_spectrumfnc(E_nu, mpv)

        def integrand_denominator(E_nu):
            return self.neutrino_spectrumfnc(E_nu, mpv)

        integral_numerator0, _ = integrate.quad(integrand0, E_min, E_max)
        integral_numerator1, _ = integrate.quad(integrand1, E_min, E_max)

        # Denominator: Integral of the neutrino spectrum (normalization factor)

        integral_denominator, _ = integrate.quad(integrand_denominator, E_min, E_max)
        crossR[0] = integral_numerator0 / integral_denominator
        crossR[1] = integral_numerator1 / integral_denominator
        crossR[2] = (crossR[0]+crossR[1])/crossR[0] # total cross section to the weak

        return crossR

    def diff_cross_section(self, E_nu, T):
        crossterm=[0,0,0,0,0]
        # Convert T to NumPy array if it's a list
        #T = np.array(T, dtype=float)

        prefactor = 2.0 * self.GF**2 * self.me
        sin2_theta_w = self.sin2_theta_w

        gl = np.array([
            0.5 * (2. * sin2_theta_w + 0.5 + 0.5),
            0.5 * (2. * sin2_theta_w - 0.5 - 0.5),
            0.5 * (2. * sin2_theta_w - 0.5 - 0.5)
        ])
        gr = np.array([
            0.5 * (2. * sin2_theta_w + 0.5 - 0.5),
            0.5 * (2. * sin2_theta_w - 0.5 + 0.5),
            0.5 * (2. * sin2_theta_w - 0.5 + 0.5)
        ])

        term_factor = (1 - T / E_nu) ** 2
        term_common = (self.me * T) / (E_nu**2 + 1e-10)  # Avoid division by zero

        crossterm[0] = gl[0]**2 + gr[0]**2 * term_factor - gl[0] * gr[0] * term_common
        crossterm[1] = gl[1]**2 + gr[1]**2 * term_factor - gl[1] * gr[1] * term_common
        crossterm[2] = gl[2]**2 + gr[2]**2 * term_factor - gl[2] * gr[2] * term_common
        crossterm[3] = prefactor * (crossterm[0] + crossterm[1] + crossterm[2])
        crossterm[4] = math.pi * (self.fine_struct / self.me)**2 * (self.moment**2) * (1.0 / T - 1.0 / E_nu)
        return crossterm
     

   
    def f(self,x):                  #used to distribute t according to 1/x
        return 1/x
        
    def inverse(self, limits, u):   #optional, used intead of "distrubutingLoop" to analytically give the 1/x distribution
        return limits[1]**u/(limits[0]**(u-1))
        
    def inverse2(self, limits, u):   #optional, used intead of "distrubutingLoop" to analytically give the 1/x**2 distribution NOT USED
        return limits[0] / np.sqrt(1 - u * (1 - (limits[0] / limits[1])))

    def kinematicFcn(self, a, b, c):
        return (a-b-c)**2-4*b*c

# Kinematics
    def tMin(self, s, gamma, vCM):
        # Converts the angleCut to tMin
        tan_angle_cut = np.tan(self.angleCut)
        tan_angle_cut_squared = tan_angle_cut ** 2

        # Precompute common terms
        square = 1 + vCM + 1 / (gamma**2 * tan_angle_cut_squared)
        l = self.kinematicFcn(s, 0, self.mSq)
    
        # Calculate triangle term
        triangle = s / l + s / (l * gamma**2 * tan_angle_cut_squared)

        # Calculate the discriminant
        discriminant = 0.25 * square**2 / triangle**2 - (l * (1 + vCM**2 + 2 * vCM)) / (4 * s * triangle)

        # Return the absolute value of the result
        return abs(-0.5 * square / triangle + np.sqrt(discriminant))

    def matrixElement(self, S, T):           #calculates the squared matrix element based on s,t MGULER
        return math.pi*(self.fine_struct/self.me)**2*(self.moment**2)*(S+T-self.me**2)/T #F.Ould-Saada paper

    def generate34(self, i, E1, E2, theta, phi):
        # Calculate p3, p4 based on quantities generated in generateT
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
    
        p3 = [E1, E1 * sin_theta * cos_phi, E1 * sin_theta * sin_phi, E1 * cos_theta]
        p4 = [E2, E1 * np.sin(math.pi - theta) * np.cos(math.pi + phi), E1 * np.sin(math.pi - theta) * np.sin(math.pi + phi), E1 * np.cos(math.pi - theta)]
    
        self.p3.append(p3)
        self.p4.append(p4)

    def boostBack(self, i, gamma, vCM, E1, E2, theta):
        # Boost p3, p4 back to LAB frame
        cos_theta = np.cos(theta)
    
        self.p3[i][0] = gamma * E1 * (1 + vCM * cos_theta)
        self.p3[i][3] = gamma * E1 * (cos_theta + vCM)
        self.p4[i][0] = gamma * (E2 - E1 * vCM * cos_theta)
        self.p4[i][3] = gamma * (vCM * E2 - cos_theta * E1)

#Below for the spectrum generation
#Use approximate function to calculate averaged differential cross sections for weak
#and magnetic moment scattering

    def neutrino_spectrumfnc(self,x,mpv):
        """
        Approximate the spectrum Landau-like distribution function.
        Args:
           x (array-like): Input data points.
           mpv (float): Most probable value (peak position).
           eta (float): Scale parameter (width).
           amplitude (float): Amplitude (normalization factor).
           Returns: array-like: Evaluated Landau function values.
        """
        eta=100
        amplitude=10.
        xi = (x - mpv) / eta
        return amplitude * np.exp(-0.5 * (xi + np.exp(-xi)))
        
    def load_data_from_file(self, file_path):
        # Load data from a text file
        data = np.loadtxt(file_path)
        if data.shape[1] != 2:
            raise ValueError("The file must contain exactly two columns: bins and counts.")
        self.bins = data[:, 0]
        self.counts = data[:, 1]
        self.neutrino_spectrum()  # Call to compute the CDF and inverse CDF after loading data

    def neutrino_spectrum(self):
        """Generate a Monte Carlo sample from a given histogram distribution."""
        probabilities = self.counts / np.sum(self.counts)  # Normalize counts to probabilities
        cdf = np.cumsum(probabilities)  # Compute cumulative distribution function
        cdf /= cdf[-1]  # Ensure CDF goes from 0 to 1

        # Use PCHIP interpolator for better monotonic inversion of the CDF
        #self.inverse_cdf = PchipInterpolator(cdf, self.bins)

        #second approach use interpredor
        # Interpolate to find the corresponding value for the random number
        self.interpolator = interp1d(cdf, self.bins, bounds_error=False, fill_value=(self.bins[0], self.bins[-1]))
        
    def generate_mc_events(self, num_mc_events):
        # Generate uniform random numbers and invert the CDF
        uniform_samples = np.random.uniform(0, 1, num_mc_events)
        # Get the continuous value corresponding to the random number using PchipInterpolator interpolator
        #mc_samples = self.inverse_cdf(uniform_samples)
       # Get the continuous value corresponding to the random number using interp1d interpolator
        mc_samples = self.interpolator(uniform_samples)

        return mc_samples

    #main method of the program. Facilitates the distribution of the initial momenta, of "t".
    #Creates the phase-space point and computes the relevant quantities such as gamma and vCM.
    #Computes the matrix element and the jacobian. Computes the outgoing momenta. Finally calcultes the resulting integrals (eg the
    #cross-section).

    def generateST(self):
        # Preallocate lists for efficiency
        t = []
        angleList = []
        indexList = np.arange(1, self.eventNum + 1)  # Directly store indices
        transverseMomentum = []
        thetaList = []
        energyList = []
        CrossSecR = []
        CrosSec = []
        Enneutrino=[]
        me_sq = self.me**2  # Store frequently used squared mass
        neutrinoM = self.generate_mc_events(self.eventNum)
        for j in range(self.eventNum):
            Enneutrino.append(neutrinoM[j])
            p1 = [neutrinoM[j], 0, 0, neutrinoM[j]]  # Incoming neutrino momentum in Lab frame
            p2 = [self.me, 0, 0, 0]  # Electron at rest
            #print("test2", neutrinoM[j])
            s = self.scalarProduct(self.addition(p1, p2), self.addition(p1, p2))
            sSq = math.sqrt(s)
            gamma = (p1[3] + math.sqrt(p2[3]**2 + me_sq)) / sSq
            vCM = abs(p1[3] + p2[3]) / (p1[3] + math.sqrt(p2[3]**2 + me_sq))
            #
            p1_0 = math.sqrt(self.kinematicFcn(s, 0, me_sq)) / (2 * sSq)
            p1[0] = p1_0
            p1[3] = p1_0
            p2_0 = (s + me_sq) / (2 * sSq)
            p2[0] = p2_0
            p2[3] = -p1_0g
            # 
            # t range determination
            tMax = 1e-4  # Fixed maximum t value
            tMin = (-s + me_sq)**2 / s  # Based on Asım Barut's paper          
            # Sample t using inverse transform sampling
            t.append(self.inverse([tMin, tMax], np.random.random()))
            jacobian = abs(tMax - tMin)
            #
            # Compute weights
            self.weights.append(t[j] * jacobian * self.matrixElement(s, t[j]) / self.kinematicFcn(s, 0, me_sq))
            # 
            # Compute scattering angle
            theta = math.acos(-2 * s * t[j] / (self.kinematicFcn(s, 0, me_sq)) + 1)
            thetaList.append(theta)
            #angleList.append( math.atan2(abs(math.sin(theta)), (gamma * math.cos(theta) + gamma * vCM)))
            #
            phi = np.random.uniform(0, 2 * math.pi)  # Generate phi randomly
            #
            # Generate final state momenta
            self.generate34(j, p1_0, p2_0, theta, phi)
            self.boostBack(j, gamma, vCM, p1_0, p2_0, theta)
            #
            # Compute energy of the electron and transverse momentum
            energyList.append(self.p4[j][0])
            transverseMomentum.append(math.sqrt(self.p4[j][1]**2 + self.p4[j][2]**2))
            #
            # calculate spectrum of electron scattering angle using Cloumb scatering formula.
            angleList.append(math.acos((neutrinoM[j] + self.me) / neutrinoM[j] * math.sqrt(self.p4[j][0] / (self.p4[j][0] + 2.0 * self.me))))
            #par_eng1 = math.cos(theta)**2 * neutrinoM[j]**2
            #par_eng2 = neutrinoM[j] + self.me / 2. + (math.sin(theta)**2 * neutrinoM[j]**2) / (2. * self.me)
            #DelecEngy = par_eng1 / par_eng2
            # Compute cross-section ratio 
            CrosSec = self.averaged_diff_cross_section(neutrinoM[j],self.p4[j][0])
            CrossSecR.append(CrosSec[2])
            #print("test",DenergyList2)
        sumT = np.sum(t)  # Summing t after loop for efficiency
        
        meanan = np.mean(angleList)
        std_devan = np.std(angleList)
        entriesan = len(angleList)
        
        meanen1 = np.mean(mc_events)
        std_deven1 = np.std(mc_events)
        entriesen1 = len(mc_events)
        
        meanen2 = np.mean(Enneutrino)
        std_deven2 = np.std(Enneutrino)
        entriesen2 = len(Enneutrino)
        
        meanen3 = np.mean(NuMaS_instance.bins)
        std_deven3 = np.std(NuMaS_instance.bins)
        entriesen3 = len(NuMaS_instance.bins)

        meaner = np.mean(energyList)
        std_dever = np.std(energyList)
        entrieser = len(energyList)

        meanan = np.mean(angleList)
        std_devan = np.std(angleList)
        entriesan = len(angleList)


        # Plot the histogram of the original data and the generated MC events
        plt.figure(figsize=(10, 6))
        plt.xlabel(r'$E_\nu$ (GeV)')
        plt.ylabel('a.u.')
        stats_text3 = f"Entries: {entriesen2}\nMean: {meanen2:.2f}\nStd Dev: {std_deven2:.2f}"
        plt.text(0.95, 0.95, stats_text3, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        plt.hist(Enneutrino, bins=100, density=False, alpha=0.6, color='r', label="Generated MC Events")
        plt.title('Tau neutrino energy spectrum')
        plt.hist(mc_events, bins=len(NuMaS_instance.bins)*10, density=True, alpha=0.6, color='r', label="Generated MC Events")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.xlabel(r'$E_\nu$ (GeV)')
        plt.ylabel('a.u.')
        plt.title('Tau neutrino energy spectrum(from distribution)')
        stats_text3 = f"Entries: {entriesen1}\nMean: {meanen1:.2f}\nStd Dev: {std_deven1:.2f}"   
        plt.text(0.95, 0.95, stats_text3, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        plt.hist(Enneutrino, bins=100, density=False, alpha=0.6, color='r', label="Generated MC Events")
        #plt.show()
        #
        plt.figure(figsize=(10, 6)) 
        plt.xlabel(r'$E_\nu$ (GeV)')
        plt.ylabel('a.u.')
        plt.title('Tau neutrino energy spectrum(given histogram)')
        plt.bar(NuMaS_instance.bins, 4*NuMaS_instance.counts/1e7, width=np.diff(NuMaS_instance.bins, prepend=NuMaS_instance.bins[0]), align='edge', alpha=0.7, color='blue', edgecolor='black')
        stats_text3 = f"Entries: {entriesen1}\nMeanen2: {meanen2:.2f}\nStd Dev: {std_deven2:.2f}"
        plt.text(0.95, 0.95, stats_text3, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        plt.hist(Enneutrino, bins=100, density=False, alpha=0.6, color='r', label="Generated MC Events")
        plt.show()
        #
        plt.title(r'Electron recoil energy')  
        plt.xlabel(r'$E_{e}$  [GeV]')
        #plt.ylabel(r'Weights [GeV$^{-2}$]')
        stats_text3 = f"Entries: {entrieser}\nMean: {meaner:.2f}\nStd Dev: {std_dever:.2f}"
        plt.text(0.95, 0.95, stats_text3, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))        
        plt.hist(energyList,bins=200, density=False, histtype='step', color='b',range=[0, 40])
        plt.savefig("Delectronenergy.png", dpi=600)
        plt.show()
        #
        plt.title(r'Scattering Angle')
        plt.xlabel(r'$\theta$ [rad]')
        plt.ylabel(r'Counts')
        stats_text2 = f"Entries: {entriesan}\nMean: {meanan:.2f}\nStd Dev: {std_devan:.2f}"
        plt.text(0.95, 0.95, stats_text2, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        plt.hist(angleList,bins=200, density=False, histtype='step', color='b',range=[0, 0.1])
        plt.savefig("electronangle.png", dpi=600)
        plt.show()

        #cross section ratio
        plt.title(r'Ratio of Total to Weak Cross Sections')
        plt.xlabel(r'${\mathrm{E_e}}$ [GeV]')
        plt.ylabel(r'Total/Weak')
        plt.loglog(energyList,CrossSecR,'ro',linewidth=2, markersize=2)
        plt.text(x=0.6, y=8, s=r"$\mu_\nu = 10^{-8}\mu_B$", fontsize=14, color='red', fontweight='bold') 
        plt.plot(2, 1.05, 'wo', markersize=10, markeredgecolor='white', markeredgewidth=1.)
        plt.plot(3, 1.04, 'wo', markersize=10, markeredgecolor='white', markeredgewidth=1.)
        plt.axhline(y=1, color='red', linestyle='dashed', linewidth=2, label="Threshold")
        #plt.xscale('log')
        plt.ylim(0, 20)
        plt.xlim(1e-3, 1000)
        plt.savefig("tCrossWT.png", dpi=600)
        plt.show()
        #
        plt.title(r'Differential Cross-Section d$\sigma/d|t|$.')
        plt.xlabel(r'$|t|$ [GeV$^2$]')
        plt.ylabel(r'Weights [GeV$^{-2}$]')
        plt.hist(t ,bins=200, density=False, histtype='step', weights=self.weights/sumT, color='k')     #plot histograms
        #savefig("diffT{}GeV.png".format(self.neutrinoArgs[0]), dpi=600)
        plt.show()
         
        plt.title(r'Transverse Momentum')
        plt.xlabel(r'p$_{\mathrm{T}}$ [GeV]') 
        plt.ylabel(r'Counts')
        plt.hist(transverseMomentum,bins=200, density=False, histtype='step', color='b')
        plt.savefig("transvMom_4GeV.png", dpi=600)
        plt.show()


        #
        with open(output_filename, "w") as file:
           for enu, elec, ang  in zip(neutrinoM, energyList, angleList):
             file.write(f"{enu:4.4f}{elec:10.4f}{ang:10.4f}\n")  # Write each float with 4 decimal places

    #def saveData(self, crossSecPlainMC, crossSecImportS):       #saves cross-sections to a .txt file
    #    with open("eventGenSavedData.txt", "a+") as f:
    #        f.write("Cross-section {} GeV (plainMC): {} mb.  ".format(self.neutrinoArgs[0],crossSecPlainMC))
    #        f.write("Cross-section {} GeV (Importance S.): {} mb. \n".format(self.neutrinoArgs[0],crossSecImportS))

# Load data from a text file
file_path = 'histogram_data3.txt'  # Replace with your actual file path
# Create an instance of NuMaSClass
            
# Generate Monte Carlo events
num_mc_events = 10000
numag=1e-8
NuMaS_instance = NuMaSClass(num_mc_events,numag)
NuMaS_instance.load_data_from_file(file_path)
            
mc_events = NuMaS_instance.generate_mc_events(num_mc_events)
NuMaS_instance.generateST()
            

