import os
import json
import tkinter as tk
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from decouple import config
from base64 import b64encode
from datetime import datetime

DISTRICTS = ['Évora', 'Leiria', 'Santarém', 'Aveiro', 'Portalegre', 'Viseu', 'Beja', 'Porto', 'Braga', 'Castelo branco',
             'Guarda', 'Faro', 'Viana do castelo', 'Bragança', 'Vila real', 'Coimbra', 'Açores', 'Lisboa', 'Madeira', 'Setúbal']

data = [[None]]


class App:
    def __init__(self, start_date: str = "01-01-2021", end_date: str = datetime.today().strftime('%d-%m-%Y')):
        self.start_date = start_date
        self.end_date = end_date
        self.headers = self.get_headers()
        self.getInfo()

    @staticmethod
    def get_headers():
        """
        Function to get the headers for the API.

        @return:
            headers: headers for the API
        """

        user = config('API_user')
        password = config('API_password')
        user_pass = '{}:{}'.format(user, password).encode('utf-8')
        b64_user_pass = b64encode(user_pass).decode('utf-8')
        headers = {'Authorization': 'Basic %s' % b64_user_pass}

        return headers

    def getInfo(self, save_json: bool = True, district: str = 'Braga') -> None:
        """
        Function to get covid-19 data from the API.

        @args:
            save_json: save json file
            district: query district

        @return:
            json: json with the data
        """

        self.district = district

        # request to the API
        r = requests.get(
            f'https://covid19-api.vost.pt/Requests/get_entry_counties/{self.start_date}_until_{self.end_date}', headers=self.headers
        )

        # if REST API isn't working, return exception
        if r.status_code != 200:
            raise Exception(f'REST API error: {r.status_code}')

        # convert r.json() to a pandas dataframe
        df = pd.DataFrame(r.json())

        # convert start_date and end_date to datetime format
        start_date = datetime.strptime(self.start_date, '%d-%m-%Y')
        end_date = datetime.strptime(self.end_date, '%d-%m-%Y')

        # convert api 'data' column to datetime format
        df['data'] = df['data'].apply(
            lambda x: datetime.strptime(x, '%d-%m-%Y'))

        # filter API data for given disctrict
        df = df[(df['distrito'] == district.upper())]

        # making sure dates are within the range requested
        df = df[(df['data'] >= self.start_date)
                & (df['data'] <= self.end_date)]

        # get only important columns from dataframe
        df = df[['data', 'concelho', 'confirmados_1']]

        # rename 'confirmados_1' column
        df = df.rename(
            columns={
                'data': 'date',
                'concelho': 'county',
                'confirmados_1': 'daily_cases'
            }
        )

        # match desired JSON format with pivot table '{<county1>: {<date1>: <daily_cases1>, ...}, ...}'
        df = df.pivot_table(index=['date'],
                            columns='county', values='daily_cases')

        # change pivot table index to string format 'dd-mm-yyyy'
        df.index = df.index.astype(str)

        # save to json file
        if save_json:
            df.to_json('data.json', indent=4,
                       force_ascii=False, orient='columns')

        self.data = df.to_dict()

    def plotData(self, county: str = None, degree: int = 6, predictions: int = 3) -> None:
        """
        Function to plot the county data usign matplotlib.

        @args:
            county: county to plot
            degree: degree of the polynomial
            predictions: number of forward month predictions

        @return:
            None
        """

        # create a new figure
        plt.figure()

        # get county 'dates' to datetime format
        dates = [datetime.strptime(d, '%Y-%m-%d')
                 for d in list(self.data[county.upper()].keys())]

        # get new dates for future predictions
        if predictions > 0:
            month_diff = 1 * 30 * 24 * 60 * 60
            last_date_ts = dates[-1].timestamp()
            new_dates_ts = list(
                range(
                    int(last_date_ts) + month_diff,
                    int(last_date_ts) + (predictions+1) * month_diff,
                    month_diff
                )
            )
            dates += [datetime.fromtimestamp(x) for x in new_dates_ts]

        # normalize county dates to serve polyfit function as x
        x = np.array(
            [0] + [x.days for x in np.diff(np.array(dates))]).cumsum()

        # get confirmed cases for each date in county data
        y = list(self.data[county].values())

        # convert lists to np.array to speed up computation
        x, y = np.array(x).astype(int), np.array(y).astype(int)

        # get polynomial coefficients and function
        if predictions > 0:
            z = np.polyfit(x[:-predictions], y, degree)
        else:
            z = np.polyfit(x, y, degree)
        p = np.poly1d(z)

        # split confirmed and predicted cases with a vertical line
        if predictions > 1:
            plt.axvline(x[-predictions-1:-predictions+1].mean(),
                        color='r', linestyle='--')
        elif predictions == 1:
            plt.axvline(x[-2:].mean(), color='r', linestyle='--')

        # plot confirmed cases graph
        if predictions > 0:
            plt.plot(x[:-predictions], y, 'o-')
        else:
            plt.plot(x, y, 'o-')

        # create random data to plot the polynomial graph
        x_aux = np.linspace(x.min(), x.max(), 100)
        plt.plot(x_aux, p(x_aux), '-')

        # plot predicted y data
        if predictions > 0:
            plt.plot(x[-predictions:], p(x[-predictions:]), '*', markersize=9)

        # convert datetime to string dd-mm-yyyy format
        x_labels = [d.strftime('%d-%m-%Y') for d in dates]

        # get x labels and convert to user friendly datetime format
        if predictions > 0:
            x_labels = x_labels[:-predictions][::3] + x_labels[-predictions:]
            x_ticks = list(x[:-predictions])[::3] + list(x[-predictions:])
        else:
            x_labels = x_labels[::3]
            x_ticks = list(x)[::3]
        plt.xticks(x_ticks, x_labels, rotation=45)

        # add legend_labels to the plot
        legend_labels = ['confirmed vs predicted split']
        legend_labels += ['confirmed covid-19 cases']
        legend_labels += [f'{degree} order polinomial graph ']
        legend_labels += ['next 3 month prediction']
        plt.legend(legend_labels)

        # add title to the plot and plot it
        plt.title(f'Covid-19 cases in {county.capitalize()}')
        plt.show()

    def startGUI(self) -> None:
        """
        Function to start user-friendly GUI to plot the data.

        @return:
            None
        """

        # create a window
        window = tk.Tk()

        # set window title
        window.title('Covid cases prediction')

        # District selection dropdown menu
        district_label = tk.Label(window, text='Select a district')
        district_menu = tk.StringVar(window)
        district_menu.set('Braga')
        district_menu_dropdown = tk.OptionMenu(
            window, district_menu, *DISTRICTS)

        # get API data from the past year
        self.getInfo(save_json=True, district=district_menu.get())

        # County selection dropdown menu
        county_label = tk.Label(window, text='Select a county')
        county_menu = tk.StringVar(window)
        county_menu.set(list(self.data.keys())[0].capitalize())
        county_menu_dropdown = tk.OptionMenu(
            window, county_menu, *list(map(lambda x: x.capitalize(), list(self.data.keys()))))

        # Degree input field
        degree_label = tk.Label(window, text='Polynomial degree?')
        degree_input = tk.Entry(window)
        degree_input.insert(0, '6')
        degree_input.config(width=4)

        # Next Predictions input field
        pred_label = tk.Label(window, text='How many predictions?')
        pred_input = tk.Entry(window)
        pred_input.insert(0, '3')
        pred_input.config(width=4)

        # create a button
        button = tk.Button(
            window,
            text='Plot selected county data',
            command=lambda: self.plotData(
                county_menu.get().upper(),
                int(degree_input.get()),
                int(pred_input.get())
            )
        )
        button.config(width=35)

        # create a button to close all the matplot windows
        close_button = tk.Button(
            window,
            text='Close all matplotlib windows',
            command=lambda: plt.close('all')
        )
        close_button.config(width=35)

        # add widgets to the window and show it
        district_label.grid(row=0, column=0)
        district_menu_dropdown.grid(row=0, column=1)
        county_label.grid(row=1, column=0)
        county_menu_dropdown.grid(row=1, column=1)
        degree_label.grid(row=2, column=0)
        degree_input.grid(row=2, column=1)
        pred_label.grid(row=3, column=0)
        pred_input.grid(row=3, column=1)
        button.grid(row=4, columnspan=2)
        close_button.grid(row=5, columnspan=2)

        # execute function on district_menu selection change
        def district_menu_change(*args):
            self.getInfo(save_json=True, district=district_menu.get())
            county_menu.set(list(self.data.keys())[0].capitalize())
            county_menu_dropdown = tk.OptionMenu(
                window, county_menu, *list(map(lambda x: x.capitalize(), list(self.data.keys()))))
            county_menu_dropdown.grid(row=1, column=1)

        # bind function to district_menu selection click
        district_menu.trace('w', district_menu_change)

        # start the main loop
        window.mainloop()


if __name__ == '__main__':
    myApp = App()  # Initialize app
    myApp.startGUI()  # Start app
