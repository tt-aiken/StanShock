#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
    Copyright 2017 Kevin Grogan

    This file is part of StanShock.

    StanShock is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License.

    StanShock is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with StanShock.  If not, see <https://www.gnu.org/licenses/>.
'''
import os.path
from typing import Optional

from StanShock.stanShock import stanShock
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import time
import cantera as ct
from numpy.polynomial.polynomial import Polynomial as poly

from StanShock.utils import getPressureData


def main(data_filename: str = "data/validation/case1.csv",
         mech_filename: str = "data/mechanisms/N2O2HeAr_justTR.xml",
         show_results: bool = True,
         results_location: Optional[str] = None) -> None:
    # =============================================================================
    # provided condtions for Case 1
    Ms = 8.4246
    # Ms = 4.0
    T1 = 296.0
    p1 = 6.666118
    # tFinal = 4.25e-3
    tFinal = 2.9e-3*8.4246/Ms
    delta = 1.0e-4  # distance to smear the initial conditions; models incomplete initial formation of shock.

    p4BLFac = 1.00

    runBLCase = False

    # plotting parameters
    fontsize = 12
    plotTimeSlice = 2.0e-3

    # provided geometry
    DDriven = 0.1524
    DDriver = DDriven
    LDriver = 3.30
    LDriven = 10.0

    # CFL settings
    cfl_nbl = 0.9
    cfl_bl = 0.9

    # Printout increment
    print_inc = 100

    # Set up gasses and determine the initial pressures
    u1 = 0.0;
    u4 = 0.0;  # initially 0 velocity
    gas1 = ct.Solution(mech_filename)
    gas4 = ct.Solution(mech_filename)
    T4 = T1;  # assumed
    gas1.TPX = T1, p1, "O2:1.00"
    gas4.TPX = T4, p1, "HE:1.00"  # use p1 as a place holder
    g1 = gas1.cp / gas1.cv
    g4 = gas4.cp / gas4.cv
    p2 = p1 * ((2*g1*Ms**2.0) - (g1-1.0)) / (g1+1.0)
    a4oa1 = np.sqrt(g4 / g1 * T4 / T1 * gas1.mean_molecular_weight / gas4.mean_molecular_weight)
    p4 = p2 * (1.0 - (g4 - 1.0) / (g1 + 1.0) / a4oa1 * (Ms - 1.0 / Ms)) ** (
                -2.0 * g4 / (g4 - 1.0))  # from handbook of shock waves
    gas4.TP = T4, p4*p4BLFac

    # set up geometry
    nX = 20000  # mesh resolution
    xProbe = 9.995
    XTSkipSteps=1
    xLower = -LDriver
    xUpper = LDriven
    xShock = 0.0
    geometry = (nX, xLower, xUpper, xShock)
    DeltaD = DDriven - DDriver
    DeltaX = (xUpper - xLower) / float(nX) * 10  # diffuse area change for numerical stability

    def D(x):
        diameter = DDriven + (DeltaD / DeltaX) * (x - xShock)
        diameter[x < (xShock - DeltaX)] = DDriver
        diameter[x > xShock] = DDriven
        return diameter

    def dDdx(x):
        dDiameterdx = np.ones(len(x)) * (DeltaD / DeltaX)
        dDiameterdx[x < (xShock - DeltaX)] = 0.0
        dDiameterdx[x > xShock] = 0.0
        return dDiameterdx

    A = lambda x: np.pi / 4.0 * D(x) ** 2.0
    dAdx = lambda x: np.pi / 2.0 * D(x) * dDdx(x)
    dlnAdx = lambda x, t: dAdx(x) / A(x)

    boundaryConditions = ['reflecting', 'reflecting']
    state1 = (gas1, u1)
    state4 = (gas4, u4)

    # set up solver parameters
    ssbl = stanShock(gas1, initializeRiemannProblem=(state4, state1, geometry),
                    boundaryConditions=boundaryConditions,
                    cfl=cfl_bl,
                    outputEvery=print_inc,
                    includeBoundaryLayerTerms=True,
                    reacting=False,
                    DOuter=D,
                    Tw=T1,  # assume wall temperature is in thermal eq. with gas
                    dlnAdx=dlnAdx,
                    includeDiffusion=False)
    ssbl.addProbe(xProbe)
    ssbl.addXTDiagram('p',skipSteps=XTSkipSteps)
    ssbl.addXTDiagram('T',skipSteps=XTSkipSteps)
    ssbl.addXTDiagram('u',skipSteps=XTSkipSteps)
    if runBLCase :
        print("Solving with boundary layer terms")
        # Solve
        t0 = time.perf_counter()
        ssbl.advanceSimulation(tFinal)
        t1 = time.perf_counter()
        print("The process took ", t1 - t0)

    # without  boundary layer model
    print("Solving without boundary layer model")
    gas1.TPX = T1, p1, "O2:1.00"
    gas4.TPX = T4, p4, "HE:1.00"
    ssnbl = stanShock(gas1, initializeRiemannProblem=(state4, state1, geometry),
                      boundaryConditions=boundaryConditions,
                      cfl=cfl_nbl,
                      outputEvery=print_inc,
                      includeBoundaryLayerTerms=False,
                      reacting=False,
                      DOuter=D,
                      Tw=T1,  # assume wall temperature is in thermal eq. with gas
                      dlnAdx=dlnAdx,
                      includeDiffusion=False,
                      Delta=delta)
    ssnbl.addProbe(xProbe)
    ssnbl.addXTDiagram("p",skipSteps=XTSkipSteps)
    ssnbl.addXTDiagram("T",skipSteps=XTSkipSteps)
    ssnbl.addXTDiagram("u",skipSteps=XTSkipSteps)

    # Solve
    t0 = time.perf_counter()
    ssnbl.advanceSimulation(tFinal)
    t1 = time.perf_counter()
    print("The process took ", t1 - t0)

    # import shock tube data
    tExp, pExp = getPressureData(data_filename)
    timeDifference = (12.211 - 8.10) / 1000.0  # difference between the test data and simulation times
    tExp += timeDifference

    # make plots of probe and XT diagrams
    plt.close("all")
    mpl.rcParams['font.size'] = fontsize
    #plt.rc('text', usetex=True)
    plt.figure(figsize=(8, 8))
    plt.plot(np.array(ssnbl.probes[0].t)*1000.0, np.array(ssnbl.probes[0].p)/133.322, 'k',
             label="$\mathrm{Without\ BL\ Model}$", linewidth=2.0)
    if runBLCase:
        plt.plot(np.array(ssbl.probes[0].t)*1000.0, np.array(ssbl.probes[0].p)/133.322, 'r',
                label="$\mathrm{With\ BL\ Model}$", linewidth=2.0)
    # plt.plot(tExp * 1000.0, pExp / 1.0e5, label="$\mathrm{Experiment}$", alpha=0.7)
    plt.axis([3.00, tFinal*1000.0, 0.0, 50])
    plt.xlabel("$t\ [\mathrm{ms}]$")
    plt.ylabel("$p\ [\mathrm{Torr}]$")
    plt.legend(loc="lower right")
    plt.tight_layout()

    # make plots of the measurement location temperature
    mpl.rcParams['font.size'] = fontsize
    plt.figure(figsize=(8, 8))
    plt.plot(np.array(ssnbl.probes[0].t)*1000.0, np.array(ssnbl.probes[0].T), 'k',
             label="$\mathrm{Without\ BL\ Model}$", linewidth=2.0)
    if runBLCase:
        plt.plot(np.array(ssbl.probes[0].t)*1000.0, np.array(ssbl.probes[0].T), 'r',
                label="$\mathrm{With\ BL\ Model}$", linewidth=2.0)
    plt.axis([3.00, tFinal*1000.0, 0.0, 10000])
    plt.xlabel("$t\ [\mathrm{ms}]$")
    plt.ylabel("$T\ [\mathrm{K}]$")
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Plot the pressure field from each simulation at a specified timestep
    nblTimeArray = np.array(ssnbl.result.time)
    nblTimeID = np.where(nblTimeArray == (nblTimeArray[nblTimeArray<=plotTimeSlice])[-1])[0][0]
    if runBLCase:
        blTimeArray = np.array(ssbl.result.time)
        blTimeID = np.where(blTimeArray == (blTimeArray[blTimeArray<=plotTimeSlice])[-1])[0][0]

    plt.figure()
    plt.plot(np.array(ssnbl.result.x), np.array(ssnbl.result.p[nblTimeID][:]), 'k', 
            label="$\mathrm{Without\ BL\ Model}$", linewidth=2.0)
    if runBLCase:
        plt.plot(np.array(ssbl.result.x), np.array(ssbl.result.p[blTimeID][:]), 'r', 
                label="$\mathrm{With\ BL\ Model}$", linewidth=2.0)
    plt.xlabel('x [m]')
    plt.ylabel('p [Pa]')
    plt.legend()
    plt.yscale("log")

    plt.figure()
    plt.plot(np.array(ssnbl.result.x), np.array(ssnbl.result.r[nblTimeID][:]), 'k', 
            label="$\mathrm{Without\ BL\ Model}$", linewidth=2.0)
    if runBLCase:
        plt.plot(np.array(ssbl.result.x), np.array(ssbl.result.r[blTimeID][:]), 'r', 
                label="$\mathrm{With\ BL\ Model}$", linewidth=2.0)
    plt.xlabel('x [m]')
    plt.ylabel('rho [kg/m3]')
    plt.legend()
    plt.yscale("log")

    plt.figure()
    plt.plot(np.array(ssnbl.result.x), np.array(ssnbl.result.T[nblTimeID][:]), 'k', 
            label="$\mathrm{Without\ BL\ Model}$", linewidth=2.0)
    if runBLCase:
        plt.plot(np.array(ssbl.result.x), np.array(ssbl.result.T[blTimeID][:]), 'r', 
                label="$\mathrm{With\ BL\ Model}$", linewidth=2.0)
    plt.xlabel('x [m]')
    plt.ylabel('T [K]')
    plt.legend()
    plt.yscale("log")

    plt.figure()
    plt.plot(np.array(ssnbl.result.x), np.array(ssnbl.result.u[nblTimeID][:]), 'k', 
            label="$\mathrm{Without\ BL\ Model}$", linewidth=2.0)
    if runBLCase:
        plt.plot(np.array(ssbl.result.x), np.array(ssbl.result.u[blTimeID][:]), 'r', 
                label="$\mathrm{With\ BL\ Model}$", linewidth=2.0)
    plt.xlabel('x [m]')
    plt.ylabel('u [m/s]')
    plt.legend()

    plt.figure()
    plt.plot(np.array(ssnbl.result.x), np.array(ssnbl.result.gamma[nblTimeID][:]), 'k', 
            label="$\mathrm{Without\ BL\ Model}$", linewidth=2.0)
    if runBLCase:
        plt.plot(np.array(ssbl.result.x), np.array(ssbl.result.gamma[blTimeID][:]), 'r', 
                label="$\mathrm{With\ BL\ Model}$", linewidth=2.0)
    plt.xlabel('x [m]')
    plt.ylabel('gamma [~]')
    plt.legend()

    # Fit the shock position and back out a velocity
    xsFitNBL = np.polyfit(ssnbl.result.time,ssnbl.result.shockPosition, deg=2)
    usFitNBL = np.polyder(xsFitNBL)
    fittedShockPositionNBL = np.polyval(xsFitNBL,ssnbl.result.time)
    ssnbl.result.shockVelocity = np.polyval(usFitNBL,ssnbl.result.time)
    if runBLCase:
        xsFitBL = np.polyfit(ssbl.result.time, ssbl.result.shockPosition, deg=2)
        usFitBL = np.polyder(xsFitBL)
        fittedShockPositionBL = np.polyval(xsFitBL,ssbl.result.time)
        ssbl.result.shockVelocity = np.polyval(usFitBL,ssbl.result.time)

    # make plots of the shock position and velocity
    plt.figure()
    plt.plot(np.array(ssnbl.result.time)*1000.0,np.array(ssnbl.result.shockPosition),'k',linewidth=2.0,
                label="$\mathrm{Without\ BL\ Model}$")
    plt.plot(np.array(ssnbl.result.time)*1000.0,fittedShockPositionNBL,'--k',linewidth=2.0,
                label="$\mathrm{Fitted, Without\ BL\ Model}$")
    if runBLCase:
        plt.plot(np.array(ssbl.result.time)*1000.0,np.array(ssbl.result.shockPosition),'r',linewidth=2.0,
                    label="$\mathrm{With\ BL\ Model}$")
        plt.plot(np.array(ssbl.result.time)*1000.0,fittedShockPositionBL,'--r',linewidth=2.0,
                    label="$\mathrm{Fitted, With\ BL\ Model}$")
    plt.xlabel("$t\ [\mathrm{ms}]$")
    plt.ylabel("$x_{shock}\ [\mathrm{m}]$")
    plt.legend()
    plt.tight_layout()


    plt.figure()
    plt.plot(np.array(ssnbl.result.time)*1000.0,np.array(ssnbl.result.shockVelocity),'k',linewidth=2.0,
        label="$\mathrm{Without\ BL\ Model}$")
    if runBLCase:
        plt.plot(np.array(ssbl.result.time)*1000.0,np.array(ssbl.result.shockVelocity),'r',linewidth=2.0,
            label="$\mathrm{With\ BL\ Model}$")
    plt.xlabel("$t\ [\mathrm{ms}]$")
    plt.ylabel("$u_{shock}\ [\mathrm{m/s}]$")
    plt.axis([0,tFinal*1000.0, -2000, 4000])
    plt.tight_layout()

    attenuationNBL = -100/ssnbl.result.x[-1] * np.log(ssnbl.result.shockVelocity[-1]/ssnbl.result.shockVelocity[0])
    print(f'Attenuation of simulation without boundary layer: {attenuationNBL:1.4f}% \n')

    # ssbl.plotXTDiagram(ssbl.XTDiagrams["p"],limits=[0,40])
    # ssnbl.plotXTDiagram(ssbl.XTDiagrams["t"],limits=[0,1.0e4])
    #ssbl.plotXTDiagram(ssbl.XTDiagrams["u"], limits=[0,2500])
    #ssnbl.plotXTDiagram(ssnbl.XTDiagrams["u"], limits=[0,2500])

    if show_results:
        plt.show()

    if results_location is not None:
        np.savez(
            os.path.join(results_location, "case1.npz"),
            pressure_with_boundary_layer=ssbl.probes[0].p,
            pressure_without_boundary_layer=ssnbl.probes[0].p,
            time_with_boundary_layer=ssbl.probes[0].t,
            time_without_boundary_layer=ssnbl.probes[0].t
        )
        plt.savefig(os.path.join(results_location, "case1.png"))


if __name__ == "__main__":
    main()
