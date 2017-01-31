#! python3
import numpy as np
import math
import matplotlib.pylab as plt
plt.rc('font', family='serif')
import warnings
import sys

class GeneralModel:
    def __init__(self, Name="SIR", tstart=0, tend=10, dt=0.01, **params):

        self._Name   = Name
        self._tstart = tstart
        self._t      = tstart
        self._dt     = dt
        self._tend   = tend
        self._x      = np.array([])
        self._dx     = np.array([])
        self._params = params

        self._Models = {
            "Kuramoto"  : self.KuramotoFlow,
            "SIR"       : self.SIRFlow,
            "VanderPol" : self.VanderPolFlow
            }

        self._UpdateMethods = {
            "Euler"      : self.UpdateEuler,
            "RungeKutta" : self.UpdateRK
            }

    def Initialize(self,x):
        self._x  = np.array(x)
        self._dx  = np.zeros_like(self._x)
        self._Time = np.arange(self._tstart,self._tend,self._dt)
        self._dim = len(self._x)
        self._XX  = np.zeros((len(self._Time),len(self._x)))
        self._dXX = np.zeros((len(self._Time),len(self._dx)))

    def KuramotoFlow(self, t, x, params):
        k   = params["k"]
        dx = np.zeros(len(x))
        N = int(len(x)/2)

        for ii in range(N):
            dx[ii] = params["w_{}".format(ii)] + k * np.sum([np.sin(tj-x[ii]) for tj in x])
        for ii in range(N,2*N):
            dx[ii] = 0
        return dx

    def PlotState(self, fignum=1, states=[],legend=[],colors=[], releaseplot=True):

        if len(states) == 0:
            warnings.warn("No state variables specified. Plotting all.")
            statesymbols = np.arange(self._dim)
        if len(colors) == 0:
            colors = [[np.random.uniform(0,1) for _ in range(3)] for _ in np.arange(self._dim)]


        if states == "All":
            numplots = self._p
        else:
            numplots = len(states)


        stateind = list(states.keys());


        plt.figure(fignum)
        plt.suptitle("Time Evolution of the "+self._Name+" Model")
        for i in range(numplots):
            if len(legend)==0:
                plt.subplot(numplots,1,i+1)
                plt.xlabel("Time")
                plt.ylabel(states[stateind[i]])
                ps = plt.plot(self._Time,self._XX[:,stateind[i]])
            else:
                ps = plt.plot(self._Time,self._XX[:,stateind[i]],
                        label="{}".format(legend[i]))


            plt.setp(ps, 'Color', colors[i], 'linewidth', 3)
            plt.grid(True)


        if len(legend)!=0:
            plt.xlabel("Time")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


        if releaseplot:
            plt.show()

    def PlotPhase(self, fignum=2, states={0:"x",1:"y"},color="k",releaseplot=True):
        if len(states) == 0:
            warnings.warn("No state variables specified. Plotting all.")
            statesymbols = np.arange(self._dim)

        stateind = list(states.keys());
        plt.figure(fignum)
        plt.suptitle("Phase Plot of the "+self._Name+" Model")
        ps = plt.plot(self._XX[:,stateind[0]],self._XX[:,stateind[1]])
        plt.setp(ps, 'Color', color, 'linewidth', 3)
        plt.xlabel(list(states.values())[0])
        plt.ylabel(list(states.values())[1])
        plt.grid(True)

        if releaseplot:
            plt.show()

    def PolarPlotState(self, fignum=3, states={"r":np.array([1]), "theta":np.array([2])}, statelabels=[],legend=[],colors=[], releaseplot=True):

        if len(states) == 0:
            warnings.warn("No state variables specified. Plotting all.")
            statelabels = np.arange(self._dim)
        if len(colors) == 0:
            colors = colors = [[np.random.uniform(0,1) for _ in range(3)] for _ in np.arange(self._dim)]

        if states == "All":
            numplots = int(self._dim/2)
        else:
            if len(legend) !=0 :
                numplots = int(len(legend))
            else:
                numplots = int(self._dim/2)

        rinds     = states["r"]
        thetainds = states["theta"]


        plt.figure(fignum)
        plt.suptitle("Time Evolution of the "+self._Name+" Model")
        for i in range(numplots):
            if len(legend)==0:
                plt.subplot(2*numplots,1,i+1)
                ps1 = plt.plot(self._Time,
                              self._XX[:,rinds[i]]*np.cos(self._XX[:,thetainds[i]]))
                plt.setp(ps1, 'Color', colors[i], 'linewidth', 3)
                plt.grid(True)
                plt.ylabel(statelabels[0])

                plt.subplot(2*numplots,1,i+2)
                ps2 = plt.plot(self._Time,
                              self._XX[:,rinds[i]]*np.sin(self._XX[:,thetainds[i]]))
                plt.setp(ps2, 'Color', colors[i], 'linewidth', 3)
                plt.ylabel(statelabels[1])
                plt.grid(True)
            else:
                plt.subplot(2,1,1)

                ps1 = plt.plot(self._Time,
                              self._XX[:,rinds[i]]*np.cos(self._XX[:,thetainds[i]]),label="{}".format(legend[i]))
                plt.grid(True)
                plt.setp(ps1, 'Color', colors[i], 'linewidth', 3)
                plt.ylabel(statelabels[0])
                plt.subplot(2,1,2)
                ps2 = plt.plot(self._Time,
                              self._XX[:,rinds[i]]*np.sin(self._XX[:,thetainds[i]]),label="{}".format(legend[i]))
                plt.grid(True)
                plt.ylabel(statelabels[1])
                plt.setp(ps2, 'Color', colors[i], 'linewidth', 3)


        plt.xlabel("Time")
        if len(legend)!=0:
            plt.legend(loc='center left', bbox_to_anchor=(1, 1.1))


        if releaseplot:
            plt.show()

    def PolarPlotPhase(self, fignum=4, states={"r":1, "theta":2}, statelabels=["x","y"],color="k",releaseplot=True):
        if len(states) == 0:
            warnings.warn("No state variables specified. Plotting all.")
            statelabels = np.arange(self._dim)

        stateind = states.keys();
        r     = self._XX[:,states["r"]    -1]
        theta = self._XX[:,states["theta"]-1]

        plt.figure(fignum)
        plt.suptitle("Polar Phase Plot of the "+self._Name+" Model")
        ps = plt.plot(r*np.cos(theta),r*np.sin(theta))
        plt.setp(ps, 'Color', color, 'linewidth', 3)
        plt.xlabel(statelabels[0])
        plt.ylabel(statelabels[1])
        plt.grid(True)

        if releaseplot:
            plt.show()

    def SetFlow(self, Flow):
        if self._Name in self._Models.keys():
            self.Flow = self._Models[self._Name]
        else:
            if Flow == None:
                raise SystemExit("There is no pre-defined flow method for the\
                 "+self._Name+" model. \n \
                 Please define a flow method and pass it to the simulate method.")
            else:
                self.Flow = Flow

    def SetUpdateMethod(self, updatemethod):
        try:
            self.Update = self._UpdateMethods[updatemethod]
        except:
            print("Update method provided is not known. Check your spelling.")
            print("Using Runge Kutta instead...")
            self.Update = self.UpdateRK

    def ShowAvailableModels(self):
        print(self._Models.keys())

    def ShowAvailableUpdateMethods(self):
        print(self._UpdateMethods.keys())

    def Simulate(self, Flow=None, UpdateMethod="RungeKutta"):
        self.SetUpdateMethod(UpdateMethod)
        self.SetFlow(Flow)
        self._dx = np.array(self.Flow(self._t, self._x, self._params))
        for ii in range(len(self._Time)):
            self.Update(ii);

    def SIRFlow(self, t, state, params):
        if len(self._params)==0:
            beta  = 4
            gamma = 1
            N     = 1
        else:
            beta  = params["beta"]
            gamma = params["gamma"]
            N     = params["N"]
        S, I , R = state[0], state[1], state[2]
        dS = -beta*I*S/N;
        dI =  beta*I*S/N - gamma*I
        dR =  gamma*I
        return np.array([dS, dI, dR])

    def UpdateEuler(self,ii):
        self._XX[ii,:] = self._x;
        self._dXX[ii,:] = self._dx;

        self._dx = self.Flow(self._t, self._x, self._params);
        self._x  = self._x + self._dt*self._dx;
        self._t  = self._t + self._dt

    def UpdateRK(self,ii):
        self._XX[ii,:] = self._x;
        self._dXX[ii,:] = self._dx;

        k1 = self.Flow(self._t, self._x, self._params)
        k2 = self.Flow(self._t+self._dt/2, self._x+k1*self._dt/2,  self._params)
        k3 = self.Flow(self._t+self._dt/2, self._x+k2*self._dt/2,  self._params)
        k4 = self.Flow(self._t+self._dt  , self._x+k2*self._dt  ,  self._params)

        self._x = self._x + (k1 + 2*k2 + 2*k3 + k4)*self._dt/6
        self._t = self._t + self._dt

    def VanderPolFlow(self, t, state, params):
        if len(self._params)==0:
            mu = 0.5
        else:
            mu = params["mu"]
        x, y = state[0], state[1]

        dx = y
        dy = mu*(1-x**2)*y - x

        return np.array([dx, dy])




if __name__ == "__main__":
    VDP = GeneralModel(Name="VanderPol", tstart=0, tend=50, dt=0.01)
    VDP.Initialize([0.9,0.1])
    VDP.Simulate()
    VDP.PlotState(fignum=1,states={1:"x",2:"y"},releaseplot=False)
    VDP.PlotPhase(fignum=2,color=[0.4,0.7,0.9])
