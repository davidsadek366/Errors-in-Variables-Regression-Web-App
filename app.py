"""
Errors in Variables Regression Models Browser Based Program
"""

#Included Libraries
import pandas as pd
from shiny import *
from shiny.types import FileInfo
import math
import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import cmdstanpy
import json
import pickle
from shiny.types import ImgData
from pathlib import Path
import dataframe_image as dfi

#Overall Graphical Layout of Interface
app_ui = ui.page_fluid(
    ui.navset_tab(

        #First Tab: Data Input
        ui.nav("Data Input", 
        ui.layout_sidebar(
            ui.panel_sidebar(
                ui.input_file("file1", "Choose CSV File with Measurements", accept=[".csv"], multiple=False),
                ui.input_checkbox("header", "Header", True),
                ui.input_file("file2", "Choose CSV File with Reference Distribution Data (Optional)", accept=[".csv"], multiple=False),
                ui.input_checkbox("header2", "Header", True)
            ),
            ui.panel_main(
                ui.output_image("contents"),
                ui.output_plot("scatter_plot"),
            )
        ),
        ),

        #Second Tab: Approximate Calculations
        ui.nav("Approximate Calculations",
                ui.input_numeric("refmean","Ref Mean", value=100),
                ui.input_numeric("refstd","Ref Standard Deviation", value=8),
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.input_checkbox("l1", "Least Squares (Red)", True),
                        ui.input_checkbox("l1U", "Least Squares Uncertainty", True),
                        ui.input_checkbox("l2", "Additive Correction (Green)", True),
                        ui.input_checkbox("l2U", "Additive Correction Uncertainty", True),
                        ui.input_checkbox("l3", "Multiplicative Correction (Purple)", True),
                        ui.input_checkbox("l3U", "Multiplicative Correction Uncertainty", True),
                    ),
                    ui.panel_main(
                        ui.output_plot("scatter_plot2")
                    )
                ),
                ui.output_text_verbatim("contents2")),

        #Third Tab: Full Bayesian Analysis
        ui.nav("Full Bayesian Analysis",
            ui.layout_sidebar( 
                ui.panel_sidebar(
                    ui.input_numeric("n_error_comps", "Number of Error Distribution Components", value=1),
                    ui.input_numeric("error_mu", "Means of Error Distribution Components", value=-0.0292672709896883),
                    ui.input_numeric("error_sigma", "Standard Deviations of Error Distribution Components", value=0.124821270414635),
                    ui.input_numeric("error_weights", "Error Distribution Component Weights", value=1),
                    ui.input_action_button("calculate", "Calculate"),
                ),
                ui.panel_main(
                    ui.output_image("stanTable", inline=True),
                    ui.output_image("plot1", inline=True),
                    ui.output_plot("plot2")
                )
            ) 
        )
    ),
)


def server(input: Inputs, output: Outputs, session: Session):
    #Reads in file and outputs describe table
    @output
    @render.image(delete_file=True)
    def contents():
        if input.file1() is None:
            return 

        f: list[FileInfo] = input.file1()
        #Converts data in csv to a dataframe
        df = pd.read_csv(f[0]["datapath"], header=0 if input.header() else None, delimiter=' ')
        
        #Outputs table describing dataframe to png file, then displays it and deletes file
        table = df.describe()
        dfi.export(table, 'tempTable.png')
        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": str(dir / "tempTable.png")}
        return img
    
    #Reads in same file and outputs scatter plot
    @output
    @render.plot
    def scatter_plot():
        if input.file1() is None:
            return 
        f: list[FileInfo] = input.file1()
        df = pd.read_csv(f[0]["datapath"], header=0 if input.header() else None, delimiter=' ')
        
        df = df[['diameter(nm)', 'intensity(arb.)', 'u_diameter(stDev,nm)','u_intensity(stDev,nm)']]
        
        #Iterates through values and takes the log of them
        for i in range(len(df['diameter(nm)'])):
            df['diameter(nm)'][i] = math.log(df['diameter(nm)'][i])

        for j in range(len(df['intensity(arb.)'])):
            df['intensity(arb.)'][j] = math.log(df['intensity(arb.)'][j])

        return df.plot.scatter(x= 'diameter(nm)',
                                   y= 'intensity(arb.)',
                                     c='blue')

    #Allows user inputs to be used
    @render.text
    def refStd():
        return str(input.refstd())
    @render.text
    def refMean():
        return str(input.refmean())

    #Reads in file, outputs scatter plot with three different lines and their uncertainties
    @output
    @render.plot
    def scatter_plot2():
        if input.file1() is None:
            return 
        f: list[FileInfo] = input.file1()
        df = pd.read_csv(f[0]["datapath"], header=0 if input.header() else None, delimiter=' ')
        
        #Converts data in csv to dataframe and separates by columns
        df = df[['diameter(nm)', 'intensity(arb.)', 'u_diameter(stDev,nm)','u_intensity(stDev,nm)']]
        
        #Calculating variables to find slopes and intercepts for plotting lines
        diameterStd = df['diameter(nm)'].std()
        diameterMean = df['diameter(nm)'].mean()
        stdU = ((diameterStd**2) - (float(refStd())**2)) ** 0.5
        meanU = diameterMean - float(refMean())

        #Taking log of all data points
        for i in range(len(df['diameter(nm)'])):
            df['diameter(nm)'][i] = math.log(df['diameter(nm)'][i])

        for j in range(len(df['intensity(arb.)'])):
            df['intensity(arb.)'][j] = math.log(df['intensity(arb.)'][j])

        #Calculating slopes and intercepts to plot
        slope, intercept = np.polyfit(df['diameter(nm)'], df['intensity(arb.)'], deg=1)
        myPlot = df.plot.scatter(x= 'diameter(nm)',
                                   y= 'intensity(arb.)',
                                     c='blue')
        
        newSlope = slope * ((diameterStd ** 2) / (float(refStd()) ** 2)) * (1 / (1 + (meanU / float(refMean()))) ** 2)
        yBar = intercept + (slope * math.log(float(refMean())))
        newIntercept = yBar - (newSlope * math.log(float(refMean())))

        xseq = np.linspace(min(df['diameter(nm)']), max(df['diameter(nm)']))

        #Plotting uncertainties
        for i in range(100):
            #Plotting Least Squares Uncertainty
            sample = df.sample(n=len(df['diameter(nm)']), replace=True)
            slope2, intercept2 = np.polyfit(sample['diameter(nm)'], sample['intensity(arb.)'], deg=1)
            if input.l1U():
                myPlot.plot(xseq, intercept2 + slope2 * xseq, color='grey', linewidth=0.5)

            #Plotting Additive Correction Uncertainty
            newSlope2 = slope2 * ((diameterStd ** 2) / (float(refStd()) ** 2)) * (1 / (1 + (meanU / float(refMean()))) ** 2)
            yBar2 = intercept2 + (slope2 * math.log(float(refMean())))
            newIntercept2 = yBar2 - (newSlope2 * math.log(float(refMean())))
            if input.l2U():
                myPlot.plot(xseq, newIntercept2 + newSlope2 * xseq, color='grey', linewidth=0.5)

            #Plotting Multiplicative Correction Uncertainty
            new_obs_var = np.var(df['diameter(nm)'])
            newSlope3 = slope2 * (new_obs_var / ((float(refStd()) ** 2) / (float(refMean()) ** 2)))
            newIntercept3 = np.mean(df['intensity(arb.)']) - newSlope3*math.log(float(refMean()))
            if input.l3U():
                myPlot.plot(xseq, newIntercept3 + newSlope3 * xseq, color='grey', linewidth=0.5)

        #Plotting Least Squares Line
        if input.l1():
            myPlot.plot(xseq, intercept + slope * xseq, color='red', linewidth=3)

        #Plotting Additive Correction Line
        if input.l2():
            myPlot.plot(xseq, newIntercept + newSlope * xseq, color='green', linewidth=3)

        #Calculating Multiplicative Correction slope and intercept
        obs_var = np.var(df['diameter(nm)'])
        slope3 = slope * (obs_var) / (float(refStd()) ** 2 / float(refMean()) ** 2)
        intercept3 = np.mean(df['intensity(arb.)']) - slope3*math.log(float(refMean()))

        #Plotting Multiplicative Correction Line
        if input.l3():
            myPlot.plot(xseq, intercept3 + slope3 * xseq, color='purple', linewidth=3)

        return myPlot
    
    @output
    @render.text
    def contents2():
        if input.file1() is None:
            return "Please upload a csv file"
        f: list[FileInfo] = input.file1()
        df = pd.read_csv(f[0]["datapath"], header=0 if input.header() else None, delimiter=' ')
        
        #Converts data in csv to dataframe and separates by columns
        df = df[['diameter(nm)', 'intensity(arb.)', 'u_diameter(stDev,nm)','u_intensity(stDev,nm)']]
        
        #Calculating Variables to find Slopes and Intercepts
        diameterStd = df['diameter(nm)'].std()
        diameterMean = df['diameter(nm)'].mean()
        stdU = ((diameterStd**2) - (float(refStd())**2)) ** 0.5
        meanU = diameterMean - float(refMean())

        #Taking log of all data points
        for i in range(len(df['diameter(nm)'])):
            df['diameter(nm)'][i] = math.log(df['diameter(nm)'][i])

        for j in range(len(df['intensity(arb.)'])):
            df['intensity(arb.)'][j] = math.log(df['intensity(arb.)'][j])

        #Calculating Slopes and Intercepts

        slope, intercept = np.polyfit(df['diameter(nm)'], df['intensity(arb.)'], deg=1)

        newSlope = slope * ((diameterStd ** 2) / (float(refStd()) ** 2)) * (1 / (1 + (meanU / float(refMean()))) ** 2)
        yBar = intercept + (slope * math.log(float(refMean())))
        newIntercept = yBar - (newSlope * math.log(float(refMean())))

        obs_var = np.var(df['diameter(nm)'])
        slope3 = slope * (obs_var / ((float(refStd()) ** 2) / (float(refMean()) ** 2)))
        intercept3 = np.mean(df['intensity(arb.)']) - slope3*math.log(float(refMean()))

        #Outputting Slopes and Intercepts based on which lines are being displayed
        myString = ""
        if input.l1():
            myString += "Least Squares Slope: " + str(round(slope, 3)) + "\nLeast Squares Intercept: " + str(round(intercept, 3))

        if input.l2():
            myString += "\nAdditive Correction Slope: " + str(round(newSlope, 3)) + "\nAdditive Correction Intercept: " + str(round(newIntercept, 3))

        if input.l3():
            myString += "\nMultiplicative Correction Slope: " + str(round(slope3, 3)) + "\nMultiplicative Correction Intercept: " + str(round(intercept3, 3))

        return myString

    @output
    @render.image(delete_file=True)
    def stanTable():
        import os
        from cmdstanpy import CmdStanModel

        if input.file1() is None:
            return 
        f: list[FileInfo] = input.file1()
        df = pd.read_csv(f[0]["datapath"], header=0 if input.header() else None, delimiter=' ')
        
        df = df[['diameter(nm)', 'intensity(arb.)', 'u_diameter(stDev,nm)','u_intensity(stDev,nm)']]

        #dictionary to write json file
        dict = {
            "n": len(df['diameter(nm)']),
            "y": df['intensity(arb.)'].to_list(),
            "x_obs": df['diameter(nm)'].to_list(),
            "ref_mu": math.log(float(refMean())),
            "ref_sigma": float(refStd()) / float(refMean()),
            "n_error_comps": int(errorComps()),
            "error_mu": [float(errorMu())],
            "error_sigma": [float(errorSigma())],
            "error_weights": [float(errorWeights())]
        }
        
        #If user uploads file, changes ref_mu and ref_std in dictionary
        if input.file2() != None:
            f2: list[FileInfo] = input.file2()
            df2 = pd.read_csv(f2[0]["datapath"], header=0 if input.header() else None)
            for column in df2.columns:
                data = df2[column].tolist()

            for i in range(len(data)):
                data[i] = math.log(float(data[i]))

            newRefMu = np.mean(data)
            newRefSigma = np.std(data)
            dict['ref_mu'] = newRefMu
            dict['ref_sigma'] = newRefSigma

            
            
        #Writes json file
        with open("temp.json", "w") as outfile:
            jsonDict = json.dumps(dict)
            outfile.write(jsonDict)

        #Outputs describe table when user hits calculate button
        if input.calculate():
            import os
            from cmdstanpy import CmdStanModel

            stan_file = os.path.join('log_lin_eiv_fix_mix.stan')
            model = CmdStanModel(stan_file=stan_file)
            data_file = os.path.join('temp.json')

            fit = model.sample(data=data_file)

            #Serializes 'fit' object to be reused later using pickle
            with open('data.pickle', 'wb') as f:
                pickle.dump(fit, f, pickle.HIGHEST_PROTOCOL)

            newdf = fit.draws_pd(vars=('beta', 'ly_sigma', 'y_int'))
            table = newdf.describe(percentiles=[0.025, 0.5, 0.975])
            dfi.export(table, 'tempTable2.png')
            dir = Path(__file__).resolve().parent
            img: ImgData = {"src": str(dir / "tempTable2.png")}
            return img

    #Allows user inputs to be used
    @render.text
    def errorComps():
        return str(input.n_error_comps())
    
    @render.text
    def errorMu():
        return str(input.error_mu())
    
    @render.text
    def errorSigma():
        return str(input.error_sigma())
    
    @render.text
    def errorWeights():
        return str(input.error_weights())
    
    @output
    @render.image(delete_file=True)
    def plot1():
        with open('data.pickle', 'rb') as f:
            fit1 = pickle.load(f)

        post_samps = az.from_cmdstanpy(
            posterior=fit1
        )
        if input.calculate():
            ax = az.plot_trace(post_samps, ["alpha", "beta", "ly_sigma"])
            plt.tight_layout()
            plt.savefig('tempPlot.png')
            dir = Path(__file__).resolve().parent
            img: ImgData = {"src": str(dir / "tempPlot.png")}
            return img
    
    @output
    @render.plot
    def plot2():
        with open('data.pickle', 'rb') as f:
            fit1 = pickle.load(f)

        post_samps = az.from_cmdstanpy(
            posterior=fit1
        )
        if input.calculate():
            ax = az.plot_trace(post_samps, ["alpha", "beta", "ly_sigma"])
            plt.tight_layout()

            assay_data = pd.read_table("coordinates.csv", sep=" ", header=0)        

            fig, ax = plt.subplots(nrows=1, ncols=1, dpi=300)
            ax.scatter(
            post_samps.posterior.new_obs[0, 0, 0],
            post_samps.posterior.new_obs[0, 0, 1],
            alpha=1,
            label="Simulated from Fitted Model"
            )
            ax.scatter(
            post_samps.posterior.new_obs[:, :, 0],
            post_samps.posterior.new_obs[:, :, 1],
            alpha=0.1, color="tab:blue"
            )
            ax.scatter(
            np.log(assay_data.iloc[:, 0]), np.log(assay_data.iloc[:, 1]),
            label="Original Measurements"
            )
            ax.set_xlim((np.round(np.log(np.min(assay_data.iloc[:, 0]))*10)/10,
                        np.round(np.log(np.max(assay_data.iloc[:, 0]))*10)/10 ))
            ax.set_xlabel("Diameter (nm)")
            ax.set_ylabel("Intensity (arb.)")
            ax.set_xticklabels(np.int64(np.round(np.exp(ax.get_xticks())/10)*10))
            ax.set_yticklabels(np.round(np.round(np.exp(ax.get_yticks())/0.05)*0.05, 2))
            ax.legend()
            return ax

app = App(app_ui, server)