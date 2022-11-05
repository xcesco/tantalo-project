# Version Wintermute 20220726

import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
#import talib as ta

from numba import jit

@jit(nopython=True)
def core(np_dayofweek, np_day, np_month, np_year,
         instrument, quantity, margin_percent, bigpointvalue, tick, direction, 
         costs_fixed, costs_variable, costs_pershares, order_type,
         np_open, np_high, np_low, np_close, np_volume, np_dates,
         np_enter_level, np_enter_rules, max_intraday_operations, 
         np_exit_rules, np_exit_rules_loss, np_exit_rules_gain,
         np_target_level, np_stop_level,
         time_exit, time_exit_loss, time_exit_gain, 
         money_stoploss, money_target,
         percent_stoploss, min_money_percent_stoploss,
         percent_target, min_money_percent_target, exit_on_entrybar, consecutive_trades):
    
    """
    Numba function of core
    """
    
    base_percent = 0
    
    # If we need all records of historical data we use complete population before use!
    mp = np.zeros(len(np_open))
    barssinceentry = np.zeros(len(np_open))
    entries_today = np.zeros(len(np_open))
    
    time_exit_rules = np.zeros(len(np_open))
    time_exit_loss_rules = np.zeros(len(np_open))
    time_exit_gain_rules = np.zeros(len(np_open))
    money_stoploss_rules = np.zeros(len(np_open))
    money_target_rules = np.zeros(len(np_open))
    percent_stoploss_rules = np.zeros(len(np_open))
    percent_target_rules = np.zeros(len(np_open))
    
    target_level_rules = np.zeros(len(np_open))
    stop_level_rules = np.zeros(len(np_open))

    shares_list = np.zeros(len(np_open))
    entry_price_list = np.zeros(len(np_open))
    exit_price_list = np.zeros(len(np_open))
    open_trade_list = np.zeros(len(np_open))
    closed_trade_list = np.zeros(len(np_open))
    open_equity_list = np.zeros(len(np_open))
    closed_equity_list = np.zeros(len(np_open))
    
    adverse_excursion_list = np.zeros(len(np_open))
    favorable_excursion_list = np.zeros(len(np_open))
    
    # If we need data just in case, we use append function when the event happens!
    operations = []
    operations_exit_date = []
    operations_entry_date = []
    operations_entry_price = []
    operations_entry_labels = []
    operations_exit_price = []
    operations_shares = []
    operations_exit_labels = []
    operations_duration = []
    operations_costs = []

    enter_label_list = ["" for i in range(len(np_open))]
    exit_label_list = ["" for i in range(len(np_open))]
    
    operations_max_adverse_excursion = []
    operations_max_favorable_excursion = []
    
    youcanenter = True # It controls maximum number of intraday operations with entries_today and max_intraday_operations
    
    isexitbar = False
    
    for i in range(len(np_open)):
        
        if consecutive_trades == True:
            my_enter_condition = i > 0 and np_enter_rules[i-1] == True and youcanenter == True and mp[i-1] == 0
        else:
            my_enter_condition = i > 0 and np_enter_rules[i-1] == True and youcanenter == True and mp[i-1] == 0 and exit_label_list[i-1] == ""

        if my_enter_condition:
            
            # Case market entry
            if order_type == "market":
                
                operations_entry_date.append(np_dates[i])
                mp[i] = 1
                
                if direction == "long":
                    enter_label_list[i] = "entry_long_market"
                else:
                    enter_label_list[i] = "entry_short_market" 
                entry_price_list[i] = np_open[i] #np_enter_level[i]
            
                if instrument == 1:
                    shares_list[i] = int(math.floor(quantity / np_close[i-1]))
                if instrument == 3:
                    shares_list[i] = quantity / np_close[i-1]
                if instrument == 2:
                    shares_list[i] = quantity
            
                operations_entry_labels.append(enter_label_list[i])
                operations_shares.append(shares_list[i])
                operations_entry_price.append(entry_price_list[i])
                
                base_percent = open_equity_list[i-1]
                
            # Case stop entry
            elif order_type == "stop":
                
                if direction == "long" :
                    np_enter_level[i] = (math.ceil(np_enter_level[i] / tick)) * tick
                    if np_enter_level[i] < 0:
                        np_enter_level[i] = np.nan
                    if np_high[i] >= np_enter_level[i]:
                        if np_open[i] <= np_enter_level[i]:
                            entry_price_list[i] = np_enter_level[i]
                        if np_open[i] > np_enter_level[i]:
                            entry_price_list[i] = np_open[i]
                        operations_entry_date.append(np_dates[i])
                        mp[i] = 1
                        enter_label_list[i] = "entry_long_stop"
            
                        if instrument == 1:
                            shares_list[i] = int(math.floor(quantity / np_close[i-1]))
                        if instrument == 3:
                            shares_list[i] = quantity / np_close[i-1]
                        if instrument == 2:
                            shares_list[i] = quantity

                        operations_entry_labels.append(enter_label_list[i])
                        operations_shares.append(shares_list[i])
                        operations_entry_price.append(entry_price_list[i])  
                        
                        base_percent = open_equity_list[i-1]
                        
                elif direction == "short" :
                    np_enter_level[i] = (math.floor(np_enter_level[i] / tick)) * tick
                    if np_enter_level[i] < 0:
                        np_enter_level[i] = np.nan
                    if np_low[i] <= np_enter_level[i]:
                        if np_open[i] >= np_enter_level[i]:
                            entry_price_list[i] = np_enter_level[i]
                        if np_open[i] < np_enter_level[i]:
                            entry_price_list[i] = np_open[i]
                        operations_entry_date.append(np_dates[i])
                        mp[i] = 1
                        enter_label_list[i] = "entry_short_stop"
            
                        if instrument == 1:
                            shares_list[i] = int(math.floor(quantity / np_close[i-1]))
                        if instrument == 3:
                            shares_list[i] = quantity / np_close[i-1]
                        if instrument == 2:
                            shares_list[i] = quantity

                        operations_entry_labels.append(enter_label_list[i])
                        operations_shares.append(shares_list[i])
                        operations_entry_price.append(entry_price_list[i]) 
                        
                        base_percent = open_equity_list[i-1]
                        
            # Case limit entry
            elif order_type == "limit":
                
                if direction == "long" :
                    np_enter_level[i] = (math.floor(np_enter_level[i] / tick)) * tick
                    if np_enter_level[i] < 0:
                        np_enter_level[i] = np.nan
                    if np_low[i] <= np_enter_level[i]:
                        if np_open[i] >= np_enter_level[i]:
                            entry_price_list[i] = np_enter_level[i]
                        if np_open[i] < np_enter_level[i]:
                            entry_price_list[i] = np_open[i]
                        operations_entry_date.append(np_dates[i])
                        mp[i] = 1
                        enter_label_list[i] = "entry_long_limit"
            
                        if instrument == 1:
                            shares_list[i] = int(math.floor(quantity / np_close[i-1]))
                        if instrument == 3:
                            shares_list[i] = quantity / np_close[i-1]
                        if instrument == 2:
                            shares_list[i] = quantity

                        operations_entry_labels.append(enter_label_list[i])
                        operations_shares.append(shares_list[i])
                        operations_entry_price.append(entry_price_list[i]) 
                        
                        base_percent = open_equity_list[i-1]
                        
                elif direction == "short" :
                    np_enter_level[i] = (math.ceil(np_enter_level[i] / tick)) * tick
                    if np_enter_level[i] < 0:
                        np_enter_level[i] = np.nan                    
                    if np_high[i] >= np_enter_level[i]:
                        if np_open[i] <= np_enter_level[i]:
                            entry_price_list[i] = np_enter_level[i]
                        if np_open[i] > np_enter_level[i]:
                            entry_price_list[i] = np_open[i]
                        operations_entry_date.append(np_dates[i])
                        mp[i] = 1
                        enter_label_list[i] = "entry_short_limit"
            
                        if instrument == 1:
                            shares_list[i] = int(math.floor(quantity / np_close[i-1]))
                        if instrument == 3:
                            shares_list[i] = quantity / np_close[i-1]
                        if instrument == 2:
                            shares_list[i] = quantity

                        operations_entry_labels.append(enter_label_list[i])
                        operations_shares.append(shares_list[i])
                        operations_entry_price.append(entry_price_list[i]) 
                        
                        base_percent = open_equity_list[i-1]
                
        else:
            enter_label = ""
            entry_price_list[i] = entry_price_list[i-1]
            shares_list[i] = shares_list[i-1]  
            
        
        #time exit has priority on other exits 'cause it would exit on open
        
        # MONEY STOPLOSS module
        
        # ****************** MONEY STOPLOSS ON ENTRY BAR [20211025] START  
        if exit_on_entrybar == True and money_stoploss != 0 and enter_label_list[i] != "" and direction == "long":
            instant_gainloss = shares_list[i] * (np_low[i] - entry_price_list[i]) * bigpointvalue           
            
            if instant_gainloss <= -money_stoploss:
                money_stoploss_rules[i] = 1
                mp[i] = 0   
                exit_price_list[i] = round(entry_price_list[i] - money_stoploss / (bigpointvalue * shares_list[i]), 2)
                exit_label_list[i] = "money_stoploss_entrybar_long"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(-money_stoploss + operations_costs[-1], 2)
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2)  
                isexitbar = True
                
        if exit_on_entrybar == True and money_stoploss != 0 and enter_label_list[i] != "" and direction == "short":
            instant_gainloss = shares_list[i] * (entry_price_list[i] - np_high[i]) * bigpointvalue           
            if instant_gainloss <= -money_stoploss:
                money_stoploss_rules[i] = 1
                mp[i] = 0   
                exit_price_list[i] = round(entry_price_list[i] + money_stoploss / (bigpointvalue * shares_list[i]), 2)
                exit_label_list[i] = "money_stoploss_entrybar_short"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(-money_stoploss + operations_costs[-1], 2)
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
        # ****************** MONEY STOPLOSS ON ENTRY BAR [20211025] END
                
        if money_stoploss != 0 and mp[i-1] == 1 and direction == "long" and\
             (np_exit_rules[i-1] == False and np_exit_rules_loss[i-1] == False and np_exit_rules_gain[i-1] == False) and\
             (time_exit == 0 or (time_exit != 0 and barssinceentry[i-1] < time_exit)) and\
             (time_exit_loss == 0 or 
             (time_exit_loss != 0 and barssinceentry[i-1] < time_exit_loss and open_trade_list[i-1] < 0)) and\
             (time_exit_gain == 0 or 
             (time_exit_gain != 0 and barssinceentry[i-1] < time_exit_gain and open_trade_list[i-1] > 0)):
            instant_gainloss = shares_list[i] * (np_low[i] - entry_price_list[i]) * bigpointvalue           
            if instant_gainloss <= -money_stoploss:
                money_stoploss_rules[i] = 1
                mp[i] = 0   
                exit_price_list[i] = round(entry_price_list[i] - money_stoploss / (bigpointvalue * shares_list[i]), 2)
                exit_label_list[i] = "money_stoploss_long"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(-money_stoploss + operations_costs[-1], 2)
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2)  
                isexitbar = True           
                
        if money_stoploss != 0 and mp[i-1] == 1 and direction == "short" and\
             (np_exit_rules[i-1] == False and np_exit_rules_loss[i-1] == False and np_exit_rules_gain[i-1] == False) and\
             (time_exit == 0 or (time_exit != 0 and barssinceentry[i-1] < time_exit)) and\
             (time_exit_loss == 0 or 
             (time_exit_loss != 0 and barssinceentry[i-1] < time_exit_loss and open_trade_list[i-1] < 0)) and\
             (time_exit_gain == 0 or 
             (time_exit_gain != 0 and barssinceentry[i-1] < time_exit_gain and open_trade_list[i-1] > 0)):
            instant_gainloss = shares_list[i] * (entry_price_list[i] - np_high[i]) * bigpointvalue           
            if instant_gainloss <= -money_stoploss:
                money_stoploss_rules[i] = 1
                mp[i] = 0   
                exit_price_list[i] = round(entry_price_list[i] + money_stoploss / (bigpointvalue * shares_list[i]), 2)
                exit_label_list[i] = "money_stoploss_short"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(-money_stoploss + operations_costs[-1], 2)
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
                
        # PERCENT STOPLOSS module
        
        # ****************** PERCENT STOPLOSS ON ENTRY BAR [20210822] START  
        if exit_on_entrybar == True and percent_stoploss != 0 and enter_label_list[i] != "" and money_stoploss_rules[i] == 0 and direction == "long":
            instant_gainloss = shares_list[i] * (np_low[i] - entry_price_list[i]) * bigpointvalue  
            if base_percent > 0:
                psl = -base_percent * (percent_stoploss / 100)
            else:
                psl = 0
            if psl > -min_money_percent_stoploss:
                psl = -min_money_percent_stoploss
            if instant_gainloss <= psl:
                percent_stoploss_rules[i] = 1
                mp[i] = 0   
                exit_price_list[i] = round(entry_price_list[i] + psl / (bigpointvalue * shares_list[i]), 2)
                exit_label_list[i] = "percent_stoploss_entrybar_long"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(psl + operations_costs[-1], 2)
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2)  
                isexitbar = True  
                
        if exit_on_entrybar == True and percent_stoploss != 0 and enter_label_list[i] != "" and money_stoploss_rules[i] == 0 and direction == "short":
            instant_gainloss = shares_list[i] * (entry_price_list[i] - np_high[i]) * bigpointvalue 
            if base_percent > 0:
                psl = -base_percent * (percent_stoploss / 100)
            else:
                psl = 0
            if psl > -min_money_percent_stoploss:
                psl = -min_money_percent_stoploss
            if instant_gainloss <= psl:
                percent_stoploss_rules[i] = 1
                mp[i] = 0   
                exit_price_list[i] = round(entry_price_list[i] - psl / (bigpointvalue * shares_list[i]), 2)
                exit_label_list[i] = "percent_stoploss_entrybar_short"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(psl + operations_costs[-1], 2)
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
        # ****************** PERCENT STOPLOSS ON ENTRY BAR [20210822] END         
        
        if percent_stoploss != 0 and mp[i-1] == 1 and money_stoploss_rules[i] == 0 and direction == "long" and\
             (np_exit_rules[i-1] == False and np_exit_rules_loss[i-1] == False and np_exit_rules_gain[i-1] == False) and\
             (time_exit == 0 or (time_exit != 0 and barssinceentry[i-1] < time_exit)) and\
             (time_exit_loss == 0 or 
             (time_exit_loss != 0 and barssinceentry[i-1] < time_exit_loss and open_trade_list[i-1] < 0)) and\
             (time_exit_gain == 0 or 
             (time_exit_gain != 0 and barssinceentry[i-1] < time_exit_gain and open_trade_list[i-1] > 0)):
            instant_gainloss = shares_list[i] * (np_low[i] - entry_price_list[i]) * bigpointvalue  
            if base_percent > 0:
                psl = -base_percent * (percent_stoploss / 100)
            else:
                psl = 0
            if psl > -min_money_percent_stoploss:
                psl = -min_money_percent_stoploss
            if instant_gainloss <= psl:
                percent_stoploss_rules[i] = 1
                mp[i] = 0   
                exit_price_list[i] = round(entry_price_list[i] + psl / (bigpointvalue * shares_list[i]), 2)
                exit_label_list[i] = "percent_stoploss_long"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(psl + operations_costs[-1], 2)
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2)  
                isexitbar = True
                
        if percent_stoploss != 0 and mp[i-1] == 1 and money_stoploss_rules[i] == 0 and direction == "short" and\
             (np_exit_rules[i-1] == False and np_exit_rules_loss[i-1] == False and np_exit_rules_gain[i-1] == False) and\
             (time_exit == 0 or (time_exit != 0 and barssinceentry[i-1] < time_exit)) and\
             (time_exit_loss == 0 or 
             (time_exit_loss != 0 and barssinceentry[i-1] < time_exit_loss and open_trade_list[i-1] < 0)) and\
             (time_exit_gain == 0 or 
             (time_exit_gain != 0 and barssinceentry[i-1] < time_exit_gain and open_trade_list[i-1] > 0)):
            instant_gainloss = shares_list[i] * (entry_price_list[i] - np_high[i]) * bigpointvalue 
            if base_percent > 0:
                psl = -base_percent * (percent_stoploss / 100)
            else:
                psl = 0
            if psl > -min_money_percent_stoploss:
                psl = -min_money_percent_stoploss
            if instant_gainloss <= psl:
                percent_stoploss_rules[i] = 1
                mp[i] = 0   
                exit_price_list[i] = round(entry_price_list[i] - psl / (bigpointvalue * shares_list[i]), 2)
                exit_label_list[i] = "percent_stoploss_short"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(psl + operations_costs[-1], 2)
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
                
        # Stop Level EXIT module
        
        # ****************** STOP LEVEL ON ENTRY BAR [20210822] START  
        if exit_on_entrybar == True and (np_stop_level[i] != 0 and np_stop_level[i] == np_stop_level[i]) and enter_label_list[i] != "" and\
           money_stoploss_rules[i] == 0 and percent_stoploss_rules[i] == 0 and direction == "long":         
            if np_low[i] <= np_stop_level[i] and np_open[i] >= np_stop_level[i]:
                if np_stop_level[i] > entry_price_list[i]:
                    exit_price_list[i] = entry_price_list[i]
                else:
                    exit_price_list[i] = np_stop_level[i]
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1], 2)
                stop_level_rules[i] = 1
                mp[i] = 0   
                exit_label_list[i] = "stop_level_entrybar_long"
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
            elif np_low[i] <= np_stop_level[i] and np_open[i] < np_stop_level[i]:
                exit_price_list[i] = np_open[i]
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1], 2)
                stop_level_rules[i] = 1
                mp[i] = 0   
                exit_label_list[i] = "stop_level_entrybar_long"
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True        
        # ****************** STOP LEVEL ON ENTRY BAR [20210822] END          
        
        if (np_stop_level[i] != 0 and np_stop_level[i] == np_stop_level[i]) and mp[i-1] == 1 and\
           money_stoploss_rules[i] == 0 and percent_stoploss_rules[i] == 0 and direction == "long" and\
             (np_exit_rules[i-1] == False and np_exit_rules_loss[i-1] == False and np_exit_rules_gain[i-1] == False) and\
             (time_exit == 0 or (time_exit != 0 and barssinceentry[i-1] < time_exit)) and\
             (time_exit_loss == 0 or 
             (time_exit_loss != 0 and barssinceentry[i-1] != time_exit_loss) or
             (time_exit_loss != 0 and barssinceentry[i-1] == time_exit_loss and open_trade_list[i-1] >= 0)) and\
             (time_exit_gain == 0 or 
             (time_exit_gain != 0 and barssinceentry[i-1] != time_exit_gain) or
             (time_exit_gain != 0 and barssinceentry[i-1] == time_exit_gain and open_trade_list[i-1] <= 0)):
             # 20220726: bug fixed on the time exits checks (previous code below)
             #(time_exit_loss == 0 or 
             #(time_exit_loss != 0 and barssinceentry[i-1] < time_exit_loss and open_trade_list[i-1] < 0)) and\
             #(time_exit_gain == 0 or 
             #(time_exit_gain != 0 and barssinceentry[i-1] < time_exit_gain and open_trade_list[i-1] > 0)): 
            
            if np_low[i] <= np_stop_level[i] and np_open[i] >= np_stop_level[i]:
                exit_price_list[i] = np_stop_level[i]
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1], 2)
                stop_level_rules[i] = 1
                mp[i] = 0   
                exit_label_list[i] = "stop_level_long"
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
            elif np_low[i] <= np_stop_level[i] and np_open[i] < np_stop_level[i]:
                exit_price_list[i] = np_open[i]
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1], 2)
                stop_level_rules[i] = 1
                mp[i] = 0   
                exit_label_list[i] = "stop_level_long"
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
                
        # ****************** STOP LEVEL ON ENTRY BAR [20210822] START  
        if exit_on_entrybar == True and (np_stop_level[i] != 0 and np_stop_level[i] == np_stop_level[i]) and enter_label_list[i] != "" and\
           money_stoploss_rules[i] == 0 and percent_stoploss_rules[i] == 0 and direction == "short":         
            if np_high[i] >= np_stop_level[i] and np_open[i] <= np_stop_level[i]:
                if np_stop_level[i] > entry_price_list[i]:
                    exit_price_list[i] = entry_price_list[i]
                else:
                    exit_price_list[i] = np_stop_level[i]
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(-shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1], 2)
                stop_level_rules[i] = 1
                mp[i] = 0   
                exit_label_list[i] = "stop_level_entrybar_short"
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
            elif np_high[i] >= np_stop_level[i] and np_open[i] > np_stop_level[i]:
                exit_price_list[i] = np_open[i]
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(-shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1], 2)
                stop_level_rules[i] = 1
                mp[i] = 0   
                exit_label_list[i] = "stop_level_entrybar_short"
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True        
        # ****************** STOP LEVEL ON ENTRY BAR [20210822] END  

        if (np_stop_level[i] != 0 and np_stop_level[i] == np_stop_level[i]) and mp[i-1] == 1 and\
           money_stoploss_rules[i] == 0 and percent_stoploss_rules[i] == 0 and direction == "short" and\
             (np_exit_rules[i-1] == False and np_exit_rules_loss[i-1] == False and np_exit_rules_gain[i-1] == False) and\
             (time_exit == 0 or (time_exit != 0 and barssinceentry[i-1] < time_exit)) and\
             (time_exit_loss == 0 or 
             (time_exit_loss != 0 and barssinceentry[i-1] != time_exit_loss) or
             (time_exit_loss != 0 and barssinceentry[i-1] == time_exit_loss and open_trade_list[i-1] >= 0)) and\
             (time_exit_gain == 0 or 
             (time_exit_gain != 0 and barssinceentry[i-1] != time_exit_gain) or
             (time_exit_gain != 0 and barssinceentry[i-1] == time_exit_gain and open_trade_list[i-1] <= 0)):
             # 20220726: bug fixed on the time exits checks (previous code below)
             #(time_exit_loss == 0 or 
             #(time_exit_loss != 0 and barssinceentry[i-1] < time_exit_loss and open_trade_list[i-1] < 0)) and\
             #(time_exit_gain == 0 or 
             #(time_exit_gain != 0 and barssinceentry[i-1] < time_exit_gain and open_trade_list[i-1] > 0)):          
            if np_high[i] >= np_stop_level[i] and np_open[i] <= np_stop_level[i]:
                exit_price_list[i] = np_stop_level[i]
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(-shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1], 2)
                stop_level_rules[i] = 1
                mp[i] = 0   
                exit_label_list[i] = "stop_level_short"
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
            elif np_high[i] >= np_stop_level[i] and np_open[i] > np_stop_level[i]:
                exit_price_list[i] = np_open[i]
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(-shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1], 2)
                stop_level_rules[i] = 1
                mp[i] = 0   
                exit_label_list[i] = "stop_level_short"
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
        
        # MONEY TARGET module
        
        # ****************** MONEY TARGET ON ENTRY BAR [20210822] START  
        if exit_on_entrybar == True and money_target != 0 and enter_label_list[i] != "" and money_stoploss_rules[i] == 0 and\
           percent_stoploss_rules[i] == 0 and stop_level_rules[i] == 0 and direction == "long":
            instant_gainloss = shares_list[i] * (np_high[i] - entry_price_list[i]) * bigpointvalue           
            if instant_gainloss >= money_target:
                money_target_rules[i] = 1
                mp[i] = 0   
                exit_price_list[i] = round(entry_price_list[i] + money_target / (bigpointvalue * shares_list[i]), 2)
                exit_label_list[i] = "money_target_entrybar_long"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(money_target + operations_costs[-1], 2)
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
                
        if exit_on_entrybar == True and money_target != 0 and enter_label_list[i] != "" and money_stoploss_rules[i] == 0 and\
           percent_stoploss_rules[i] == 0 and stop_level_rules[i] == 0 and direction == "short":
            instant_gainloss = shares_list[i] * (entry_price_list[i] - np_low[i]) * bigpointvalue           
            if instant_gainloss >= money_target:
                money_target_rules[i] = 1
                mp[i] = 0   
                exit_price_list[i] = round(entry_price_list[i] - money_target / (bigpointvalue * shares_list[i]), 2)
                exit_label_list[i] = "money_target_entrybar_short"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(money_target + operations_costs[-1], 2)
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2)
                isexitbar = True
        # ****************** MONEY TARGET ON ENTRY BAR [20210822] END                 
                
        if money_target != 0 and mp[i-1] == 1 and money_stoploss_rules[i] == 0 and\
           percent_stoploss_rules[i] == 0 and stop_level_rules[i] == 0 and direction == "long" and\
             (np_exit_rules[i-1] == False and np_exit_rules_loss[i-1] == False and np_exit_rules_gain[i-1] == False) and\
             (time_exit == 0 or (time_exit != 0 and barssinceentry[i-1] < time_exit)) and\
             (time_exit_loss == 0 or 
             (time_exit_loss != 0 and barssinceentry[i-1] != time_exit_loss) or
             (time_exit_loss != 0 and barssinceentry[i-1] == time_exit_loss and open_trade_list[i-1] >= 0)) and\
             (time_exit_gain == 0 or 
             (time_exit_gain != 0 and barssinceentry[i-1] != time_exit_gain) or
             (time_exit_gain != 0 and barssinceentry[i-1] == time_exit_gain and open_trade_list[i-1] <= 0)):
             # 20220726: bug fixed on the time exits checks (previous code below)
             #(time_exit_loss == 0 or 
             #(time_exit_loss != 0 and barssinceentry[i-1] < time_exit_loss and open_trade_list[i-1] < 0)) and\
             #(time_exit_gain == 0 or 
             #(time_exit_gain != 0 and barssinceentry[i-1] < time_exit_gain and open_trade_list[i-1] > 0)): 
            instant_gainloss = shares_list[i] * (np_high[i] - entry_price_list[i]) * bigpointvalue           
            if instant_gainloss >= money_target:
                money_target_rules[i] = 1
                mp[i] = 0   
                exit_price_list[i] = round(entry_price_list[i] + money_target / (bigpointvalue * shares_list[i]), 2)
                exit_label_list[i] = "money_target_long"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(money_target + operations_costs[-1], 2)
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
                
        if money_target != 0 and mp[i-1] == 1 and money_stoploss_rules[i] == 0 and\
           percent_stoploss_rules[i] == 0 and stop_level_rules[i] == 0 and direction == "short" and\
             (np_exit_rules[i-1] == False and np_exit_rules_loss[i-1] == False and np_exit_rules_gain[i-1] == False) and\
             (time_exit == 0 or (time_exit != 0 and barssinceentry[i-1] < time_exit)) and\
             (time_exit_loss == 0 or 
             (time_exit_loss != 0 and barssinceentry[i-1] != time_exit_loss) or
             (time_exit_loss != 0 and barssinceentry[i-1] == time_exit_loss and open_trade_list[i-1] >= 0)) and\
             (time_exit_gain == 0 or 
             (time_exit_gain != 0 and barssinceentry[i-1] != time_exit_gain) or
             (time_exit_gain != 0 and barssinceentry[i-1] == time_exit_gain and open_trade_list[i-1] <= 0)):
             # 20220726: bug fixed on the time exits checks (previous code below)
             #(time_exit_loss == 0 or 
             #(time_exit_loss != 0 and barssinceentry[i-1] < time_exit_loss and open_trade_list[i-1] < 0)) and\
             #(time_exit_gain == 0 or 
             #(time_exit_gain != 0 and barssinceentry[i-1] < time_exit_gain and open_trade_list[i-1] > 0)): 
            instant_gainloss = shares_list[i] * (entry_price_list[i] - np_low[i]) * bigpointvalue           
            if instant_gainloss >= money_target:
                money_target_rules[i] = 1
                mp[i] = 0   
                exit_price_list[i] = round(entry_price_list[i] - money_target / (bigpointvalue * shares_list[i]), 2)
                exit_label_list[i] = "money_target_short"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(money_target + operations_costs[-1], 2)
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2)
                isexitbar = True
                
        # PERCENT TARGET module

        # ****************** PERCENT TARGET ON ENTRY BAR [20210822] START  
        if exit_on_entrybar == True and percent_target != 0 and enter_label_list[i] != "" and money_stoploss_rules[i] == 0 and\
            percent_stoploss_rules[i] == 0 and stop_level_rules[i] == 0 and money_target_rules[i] == 0 and\
            direction == "long":
            instant_gainloss = shares_list[i] * (np_high[i] - entry_price_list[i]) * bigpointvalue  
            if base_percent > 0:
                pt = base_percent * (percent_target / 100)
            else:
                pt = 0
            if pt < min_money_percent_target:
                pt = min_money_percent_target
            if instant_gainloss >= pt:
                percent_target_rules[i] = 1
                mp[i] = 0   
                exit_price_list[i] = round(entry_price_list[i] + pt / (bigpointvalue * shares_list[i]), 2)
                exit_label_list[i] = "percent_target_entrybar_long"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(pt + operations_costs[-1], 2)
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
                
        if exit_on_entrybar == True and percent_target != 0 and enter_label_list[i] != "" and money_stoploss_rules[i] == 0 and\
           percent_stoploss_rules[i] == 0 and stop_level_rules[i] == 0 and money_target_rules[i] == 0 and\
           direction == "short":
            instant_gainloss = shares_list[i] * (entry_price_list[i] - np_low[i]) * bigpointvalue 
            if base_percent > 0:
                pt = base_percent * (percent_target / 100)
            else:
                pt = 0
            if pt < min_money_percent_target:
                pt = min_money_percent_target
            if instant_gainloss >= pt:
                percent_target_rules[i] = 1
                mp[i] = 0   
                exit_price_list[i] = round(entry_price_list[i] - pt / (bigpointvalue * shares_list[i]), 2)
                exit_label_list[i] = "percent_target_entrybar_short"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(pt + operations_costs[-1], 2)
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
        # ****************** PERCENT TARGET ON ENTRY BAR [20210822] END                
                
        if percent_target != 0 and mp[i-1] == 1 and money_stoploss_rules[i] == 0 and\
            percent_stoploss_rules[i] == 0 and stop_level_rules[i] == 0 and money_target_rules[i] == 0 and\
            direction == "long" and\
             (np_exit_rules[i-1] == False and np_exit_rules_loss[i-1] == False and np_exit_rules_gain[i-1] == False) and\
             (time_exit == 0 or (time_exit != 0 and barssinceentry[i-1] < time_exit)) and\
             (time_exit_loss == 0 or 
             (time_exit_loss != 0 and barssinceentry[i-1] != time_exit_loss) or
             (time_exit_loss != 0 and barssinceentry[i-1] == time_exit_loss and open_trade_list[i-1] >= 0)) and\
             (time_exit_gain == 0 or 
             (time_exit_gain != 0 and barssinceentry[i-1] != time_exit_gain) or
             (time_exit_gain != 0 and barssinceentry[i-1] == time_exit_gain and open_trade_list[i-1] <= 0)):
             # 20220726: bug fixed on the time exits checks (previous code below)
             #(time_exit_loss == 0 or 
             #(time_exit_loss != 0 and barssinceentry[i-1] < time_exit_loss and open_trade_list[i-1] < 0)) and\
             #(time_exit_gain == 0 or 
             #(time_exit_gain != 0 and barssinceentry[i-1] < time_exit_gain and open_trade_list[i-1] > 0)): 
            instant_gainloss = shares_list[i] * (np_high[i] - entry_price_list[i]) * bigpointvalue  
            if base_percent > 0:
                pt = base_percent * (percent_target / 100)
            else:
                pt = 0
            if pt < min_money_percent_target:
                pt = min_money_percent_target
            if instant_gainloss >= pt:
                percent_target_rules[i] = 1
                mp[i] = 0   
                exit_price_list[i] = round(entry_price_list[i] + pt / (bigpointvalue * shares_list[i]), 2)
                exit_label_list[i] = "percent_target_long"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(pt + operations_costs[-1], 2)
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
                
        if percent_target != 0 and mp[i-1] == 1 and money_stoploss_rules[i] == 0 and\
           percent_stoploss_rules[i] == 0 and stop_level_rules[i] == 0 and money_target_rules[i] == 0 and\
           direction == "short" and\
             (np_exit_rules[i-1] == False and np_exit_rules_loss[i-1] == False and np_exit_rules_gain[i-1] == False) and\
             (time_exit == 0 or (time_exit != 0 and barssinceentry[i-1] < time_exit)) and\
             (time_exit_loss == 0 or 
             (time_exit_loss != 0 and barssinceentry[i-1] != time_exit_loss) or
             (time_exit_loss != 0 and barssinceentry[i-1] == time_exit_loss and open_trade_list[i-1] >= 0)) and\
             (time_exit_gain == 0 or 
             (time_exit_gain != 0 and barssinceentry[i-1] != time_exit_gain) or
             (time_exit_gain != 0 and barssinceentry[i-1] == time_exit_gain and open_trade_list[i-1] <= 0)):
             # 20220726: bug fixed on the time exits checks (previous code below)
             #(time_exit_loss == 0 or 
             #(time_exit_loss != 0 and barssinceentry[i-1] < time_exit_loss and open_trade_list[i-1] < 0)) and\
             #(time_exit_gain == 0 or 
             #(time_exit_gain != 0 and barssinceentry[i-1] < time_exit_gain and open_trade_list[i-1] > 0)): 
            instant_gainloss = shares_list[i] * (entry_price_list[i] - np_low[i]) * bigpointvalue 
            if base_percent > 0:
                pt = base_percent * (percent_target / 100)
            else:
                pt = 0
            if pt < min_money_percent_target:
                pt = min_money_percent_target
            if instant_gainloss >= pt:
                percent_target_rules[i] = 1
                mp[i] = 0   
                exit_price_list[i] = round(entry_price_list[i] - pt / (bigpointvalue * shares_list[i]), 2)
                exit_label_list[i] = "percent_target_short"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(pt + operations_costs[-1], 2)
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
                
        # Target Level EXIT module
        
        # ****************** TARGET LEVEL ON ENTRY BAR [20210822] START  
        if exit_on_entrybar == True and (np_target_level[i] != 0 and np_target_level[i] == np_target_level[i]) and enter_label_list[i] != "" and\
           money_stoploss_rules[i] == 0 and percent_stoploss_rules[i] == 0 and stop_level_rules[i] == 0 and\
           money_target_rules[i] == 0 and percent_target_rules[i] == 0 and direction == "long":         
            if np_high[i] >= np_target_level[i] and np_open[i] <= np_target_level[i]:
                #exit_price_list[i] = np_target_level[i]
                if np_target_level[i] < entry_price_list[i]:
                    exit_price_list[i] = entry_price_list[i]
                else:
                    exit_price_list[i] = np_target_level[i] 
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1], 2)
                target_level_rules[i] = 1
                mp[i] = 0   
                exit_label_list[i] = "target_level_entrybar_long"
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
            elif np_high[i] >= np_target_level[i] and np_open[i] > np_target_level[i]:
                exit_price_list[i] = np_open[i]
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1], 2)
                target_level_rules[i] = 1
                mp[i] = 0   
                exit_label_list[i] = "target_level_entrybar_long"
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True       
        # ****************** TARGET LEVEL ON ENTRY BAR [20210822] END
        
        if (np_target_level[i] != 0 and np_target_level[i] == np_target_level[i]) and mp[i-1] == 1 and\
           money_stoploss_rules[i] == 0 and percent_stoploss_rules[i] == 0 and stop_level_rules[i] == 0 and\
           money_target_rules[i] == 0 and percent_target_rules[i] == 0 and direction == "long" and\
             (np_exit_rules[i-1] == False and np_exit_rules_loss[i-1] == False and np_exit_rules_gain[i-1] == False) and\
             (time_exit == 0 or (time_exit != 0 and barssinceentry[i-1] < time_exit)) and\
             (time_exit_loss == 0 or 
             (time_exit_loss != 0 and barssinceentry[i-1] != time_exit_loss) or
             (time_exit_loss != 0 and barssinceentry[i-1] == time_exit_loss and open_trade_list[i-1] >= 0)) and\
             (time_exit_gain == 0 or 
             (time_exit_gain != 0 and barssinceentry[i-1] != time_exit_gain) or
             (time_exit_gain != 0 and barssinceentry[i-1] == time_exit_gain and open_trade_list[i-1] <= 0)):
             # 20220726: bug fixed on the time exits checks (previous code below)
             #(time_exit_loss == 0 or 
             #(time_exit_loss != 0 and barssinceentry[i-1] < time_exit_loss and open_trade_list[i-1] < 0)) and\
             #(time_exit_gain == 0 or 
             #(time_exit_gain != 0 and barssinceentry[i-1] < time_exit_gain and open_trade_list[i-1] > 0)):          
            if np_high[i] >= np_target_level[i] and np_open[i] <= np_target_level[i]:
                exit_price_list[i] = np_target_level[i]
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1], 2)
                target_level_rules[i] = 1
                mp[i] = 0   
                exit_label_list[i] = "target_level_long"
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
            elif np_high[i] >= np_target_level[i] and np_open[i] > np_target_level[i]:
                exit_price_list[i] = np_open[i]
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1], 2)
                target_level_rules[i] = 1
                mp[i] = 0   
                exit_label_list[i] = "target_level_long"
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
                     
        # ****************** TARGET LEVEL ON ENTRY BAR [20210822] START  
        if exit_on_entrybar == True and (np_target_level[i] != 0 and np_target_level[i] == np_target_level[i]) and enter_label_list[i] != "" and\
             money_stoploss_rules[i] == 0 and percent_stoploss_rules[i] == 0 and stop_level_rules[i] == 0 and\
             money_target_rules[i] == 0 and percent_target_rules[i] == 0 and direction == "short":         
            if np_low[i] <= np_target_level[i] and np_open[i] >= np_target_level[i]:
                exit_price_list[i] = np_target_level[i]
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(shares_list[i] * (entry_price_list[i] - exit_price_list[i]) * bigpointvalue + operations_costs[-1], 2)
                target_level_rules[i] = 1
                mp[i] = 0   
                exit_label_list[i] = "target_level_entrybar_short"
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
            elif np_low[i] <= np_target_level[i] and np_open[i] < np_target_level[i]:
                exit_price_list[i] = np_open[i]
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(shares_list[i] * (entry_price_list[i] - exit_price_list[i]) * bigpointvalue + operations_costs[-1], 2)
                target_level_rules[i] = 1
                mp[i] = 0   
                exit_label_list[i] = "target_level_entrybar_short"
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True       
        # ****************** TARGET LEVEL ON ENTRY BAR [20210822] END 
        
        if (np_target_level[i] != 0 and np_target_level[i] == np_target_level[i]) and mp[i-1] == 1 and\
             money_stoploss_rules[i] == 0 and percent_stoploss_rules[i] == 0 and stop_level_rules[i] == 0 and\
             money_target_rules[i] == 0 and percent_target_rules[i] == 0 and direction == "short" and\
             (np_exit_rules[i-1] == False and np_exit_rules_loss[i-1] == False and np_exit_rules_gain[i-1] == False) and\
             (time_exit == 0 or (time_exit != 0 and barssinceentry[i-1] < time_exit)) and\
             (time_exit_loss == 0 or 
             (time_exit_loss != 0 and barssinceentry[i-1] != time_exit_loss) or
             (time_exit_loss != 0 and barssinceentry[i-1] == time_exit_loss and open_trade_list[i-1] >= 0)) and\
             (time_exit_gain == 0 or 
             (time_exit_gain != 0 and barssinceentry[i-1] != time_exit_gain) or
             (time_exit_gain != 0 and barssinceentry[i-1] == time_exit_gain and open_trade_list[i-1] <= 0)):
             # 20220726: bug fixed on the time exits checks (previous code below)
             #(time_exit_loss == 0 or 
             #(time_exit_loss != 0 and barssinceentry[i-1] < time_exit_loss and open_trade_list[i-1] < 0)) and\
             #(time_exit_gain == 0 or 
             #(time_exit_gain != 0 and barssinceentry[i-1] < time_exit_gain and open_trade_list[i-1] > 0)):  
            
            if np_low[i] <= np_target_level[i] and np_open[i] >= np_target_level[i]:
                exit_price_list[i] = np_target_level[i]
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(shares_list[i] * (entry_price_list[i] - exit_price_list[i]) * bigpointvalue + operations_costs[-1], 2)
                target_level_rules[i] = 1
                mp[i] = 0   
                exit_label_list[i] = "target_level_short"
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
            elif np_low[i] <= np_target_level[i] and np_open[i] < np_target_level[i] :
                exit_price_list[i] = np_open[i]
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(shares_list[i] * (entry_price_list[i] - exit_price_list[i]) * bigpointvalue + operations_costs[-1], 2)
                target_level_rules[i] = 1
                mp[i] = 0   
                exit_label_list[i] = "target_level_short"
                open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2) 
                isexitbar = True
        
        # EXIT module
        if np_exit_rules[i-1] == True and mp[i-1] == 1 and\
           money_stoploss_rules[i] == 0 and percent_stoploss_rules[i] == 0 and stop_level_rules[i] == 0 and\
           money_target_rules[i] == 0 and percent_target_rules[i] == 0 and target_level_rules[i] == 0:
            mp[i] = 0   
            exit_price_list[i] = np_open[i]
            if direction == "long":
                exit_label_list[i] = "exit_rules_long"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1],2)
                open_trade_list[i] = closed_trade_list[i]
            else:
                exit_label_list[i] = "exit_rules_short"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(-shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1],2)
                open_trade_list[i] = closed_trade_list[i]
            operations.append(closed_trade_list[i])
            operations_exit_date.append(np_dates[i])
            operations_exit_price.append(exit_price_list[i])
            operations_exit_labels.append(exit_label_list[i])
            closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2)
            isexitbar = True
            
        # EXIT LOSS module
        if np_exit_rules_loss[i-1] == True and mp[i-1] == 1 and open_trade_list[i-1] < 0 and\
           np_exit_rules[i-1] == False and\
           money_stoploss_rules[i] == 0 and percent_stoploss_rules[i] == 0 and stop_level_rules[i] == 0 and\
           money_target_rules[i] == 0 and percent_target_rules[i] == 0 and target_level_rules[i] == 0:
            mp[i] = 0   
            exit_price_list[i] = np_open[i]
            if direction == "long":
                exit_label_list[i] = "exit_rules_loss_long"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1],2)
                open_trade_list[i] = closed_trade_list[i]
            else:
                exit_label_list[i] = "exit_rules_loss_short"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(-shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1],2)
                open_trade_list[i] = closed_trade_list[i]
            operations.append(closed_trade_list[i])
            operations_exit_date.append(np_dates[i])
            operations_exit_price.append(exit_price_list[i])
            operations_exit_labels.append(exit_label_list[i])
            closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2)
            isexitbar = True
            
        # EXIT GAIN module
        if np_exit_rules_gain[i-1] == True and mp[i-1] == 1 and open_trade_list[i-1] > 0 and\
           np_exit_rules[i-1] == False and\
           (np_exit_rules_loss[i-1] == False or (np_exit_rules_loss[i-1] == True and open_trade_list[i-1] > 0)) and\
           money_stoploss_rules[i] == 0 and percent_stoploss_rules[i] == 0 and stop_level_rules[i] == 0 and\
           money_target_rules[i] == 0 and percent_target_rules[i] == 0 and target_level_rules[i] == 0:
            mp[i] = 0   
            exit_price_list[i] = np_open[i]
            if direction == "long":
                exit_label_list[i] = "exit_rules_gain_long"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1],2)
                open_trade_list[i] = closed_trade_list[i]
            else:
                exit_label_list[i] = "exit_rules_gain_short"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(-shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1],2)
                open_trade_list[i] = closed_trade_list[i]
            operations.append(closed_trade_list[i])
            operations_exit_date.append(np_dates[i])
            operations_exit_price.append(exit_price_list[i])
            operations_exit_labels.append(exit_label_list[i])
            closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2)
            isexitbar = True
            
        # TIME EXIT module
        if time_exit != 0 and barssinceentry[i-1] >= time_exit and mp[i-1] == 1 and\
            np_exit_rules[i-1] == False and\
           (np_exit_rules_loss[i-1] == False or (np_exit_rules_loss[i-1] == True and open_trade_list[i-1] >= 0)) and\
           (np_exit_rules_gain[i-1] == False or (np_exit_rules_gain[i-1] == True and open_trade_list[i-1] <= 0)) and\
            money_stoploss_rules[i] == 0 and percent_stoploss_rules[i] == 0 and stop_level_rules[i] == 0 and\
            money_target_rules[i] == 0 and percent_target_rules[i] == 0 and target_level_rules[i] == 0:
            time_exit_rules[i] = 1
            mp[i] = 0   
            exit_price_list[i] = np_open[i]
            if direction == "long":
                exit_label_list[i] = "time_exit_rules_long"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1],2)
                open_trade_list[i] = closed_trade_list[i]
            else:
                exit_label_list[i] = "time_exit_rules_short"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(-shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1],2)
                open_trade_list[i] = closed_trade_list[i]
            operations.append(closed_trade_list[i])
            operations_exit_date.append(np_dates[i])
            operations_exit_price.append(exit_price_list[i])
            operations_exit_labels.append(exit_label_list[i])
            closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2)
            isexitbar = True
        
        # TIME EXIT LOSS module
        if time_exit_loss != 0 and barssinceentry[i-1] >= time_exit_loss and open_trade_list[i-1] < 0 and\
            mp[i-1] == 1 and time_exit_rules[i] == 0 and\
            np_exit_rules[i-1] == False and\
           (np_exit_rules_loss[i-1] == False or (np_exit_rules_loss[i-1] == True and open_trade_list[i-1] >= 0)) and\
           (np_exit_rules_gain[i-1] == False or (np_exit_rules_gain[i-1] == True and open_trade_list[i-1] <= 0)) and\
            money_stoploss_rules[i] == 0 and percent_stoploss_rules[i] == 0 and stop_level_rules[i] == 0 and\
            money_target_rules[i] == 0 and percent_target_rules[i] == 0 and target_level_rules[i] == 0:
            time_exit_loss_rules[i] = 1
            mp[i] = 0   
            exit_price_list[i] = np_open[i]
            if direction == "long":
                exit_label_list[i] = "time_exit_rules_loss_long"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1],2)
                open_trade_list[i] = closed_trade_list[i]
            else:
                exit_label_list[i] = "time_exit_rules_loss_short"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(-shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1],2)
                open_trade_list[i] = closed_trade_list[i]
            operations.append(closed_trade_list[i])
            operations_exit_date.append(np_dates[i])
            operations_exit_price.append(exit_price_list[i])
            operations_exit_labels.append(exit_label_list[i])
            closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2)
            isexitbar = True
        
        # TIME EXIT GAIN module
        if time_exit_gain != 0 and barssinceentry[i-1] >= time_exit_gain and open_trade_list[i-1] > 0 and\
            mp[i-1] == 1 and time_exit_rules[i] == 0 and time_exit_loss_rules[i] == 0 and\
            np_exit_rules[i-1] == False and\
           (np_exit_rules_loss[i-1] == False or (np_exit_rules_loss[i-1] == True and open_trade_list[i-1] >= 0)) and\
           (np_exit_rules_gain[i-1] == False or (np_exit_rules_gain[i-1] == True and open_trade_list[i-1] <= 0)) and\
            money_stoploss_rules[i] == 0 and percent_stoploss_rules[i] == 0 and stop_level_rules[i] == 0 and\
            money_target_rules[i] == 0 and percent_target_rules[i] == 0 and target_level_rules[i] == 0:
            time_exit_gain_rules[i] = 1
            mp[i] = 0   
            exit_price_list[i] = np_open[i]
            if direction == "long":
                exit_label_list[i] = "time_exit_rules_gain_long"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1],2)
                open_trade_list[i] = closed_trade_list[i]
            else:
                exit_label_list[i] = "time_exit_rules_gain_short"
                operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                closed_trade_list[i] = round(-shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1],2)
                open_trade_list[i] = closed_trade_list[i]
            operations.append(closed_trade_list[i])
            operations_exit_date.append(np_dates[i])
            operations_exit_price.append(exit_price_list[i])
            operations_exit_labels.append(exit_label_list[i])
            closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2)
            isexitbar = True
            
        # Propagation of closed_equity_list
        if exit_label_list[i] == "":
            closed_equity_list[i] = closed_equity_list[i-1]
            
        # Propagation of mp
        if mp[i-1] == 1 and exit_label_list[i-1] == "" and exit_label_list[i] == "":
            mp[i] = 1
            
        # Barsinceentry definition
        if mp[i] == 1:
            barssinceentry[i] = barssinceentry[i-1] + 1
            
        # cosmetic action to clean the log
        if mp[i] == 0 and enter_label_list[i] == "" and exit_label_list[i] == "":
            shares_list[i] = 0
            entry_price_list[i] = 0
            
        # During the trade on current bar
        if mp[i] == 1 and exit_label_list[i] == "":
            if direction == "long":
                open_trade_list[i] = round(shares_list[i] * (np_close[i] - entry_price_list[i]) * bigpointvalue, 2)
            else:
                open_trade_list[i] = round(-shares_list[i] * (np_close[i] - entry_price_list[i]) * bigpointvalue, 2)

        if closed_trade_list[i] == 0:
            open_equity_list[i] = round(open_trade_list[i] + closed_equity_list[i],2)
        else:
            open_equity_list[i] = closed_equity_list[i]
            
        # Virtual close of last open trade if started before last bar!
        if i == len(np_open) - 1:
            if mp[i-1] == 1 and exit_label_list[i] == "":
                mp[i] = 1  
                exit_price_list[i] = np_close[i]
                if direction == "long":
                    exit_label_list[i] = "last_open_position_exit_rules_long"
                    operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                    closed_trade_list[i] = round(shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1],2)
                    open_trade_list[i] = closed_trade_list[i]
                else:
                    exit_label_list[i] = "last_open_position_exit_rules_short"
                    operations_costs.append(- 2 * costs_fixed - 2 * costs_variable / 100 * shares_list[i] * np_close[i-1] - 2 * costs_pershares * shares_list[i])
                    closed_trade_list[i] = round(-shares_list[i] * (exit_price_list[i] - entry_price_list[i]) * bigpointvalue + operations_costs[-1],2)
                    open_trade_list[i] = closed_trade_list[i]
                operations.append(closed_trade_list[i])
                operations_exit_date.append(np_dates[i])
                operations_exit_price.append(exit_price_list[i])
                operations_exit_labels.append(exit_label_list[i])
                #operations_duration.append(barssinceentry[i])
                closed_equity_list[i] = round(closed_equity_list[i-1] + closed_trade_list[i],2)
            
        # definition of number of trades per day
        if np_day[i] != np_day[i-1]:
            entries_today[i] = 0
        #elif mp[i] == 1 and mp[i-1] == 0:
        elif enter_label_list[i] != "":
            entries_today[i] = entries_today[i-1] + 1
        else:
            entries_today[i] = entries_today[i-1]
        if entries_today[i] < max_intraday_operations:
            youcanenter = True
        else:
            youcanenter = False
            
        # Maximum Adverse Excursion & Maximum Favorable Excursion
        if mp[i] == 1 or exit_label_list[i] != "":
            
            if open_trade_list[i] < 0 and open_trade_list[i] < adverse_excursion_list[i-1]:
                adverse_excursion_list[i] = open_trade_list[i]
            else:
                adverse_excursion_list[i] = adverse_excursion_list[i-1]
                
            if open_trade_list[i] > 0 and open_trade_list[i] > favorable_excursion_list[i-1]:
                favorable_excursion_list[i] = open_trade_list[i]
            else:
                favorable_excursion_list[i] = favorable_excursion_list[i-1]
                
            # On first trade bar
            if mp[i] == 1 and mp[i-1] == 0:
                if open_trade_list[i] < 0:
                    adverse_excursion_list[i] = open_trade_list[i]
                elif open_trade_list[i] > 0:
                    favorable_excursion_list[i] = open_trade_list[i]
                
        if exit_label_list[i] != "":
            operations_max_adverse_excursion.append(adverse_excursion_list[i])
            operations_max_favorable_excursion.append(favorable_excursion_list[i])
            operations_duration.append(barssinceentry[i-1])
    
    return operations, operations_exit_date, operations_entry_date,\
           operations_shares,operations_entry_labels,\
           operations_entry_price, operations_exit_price,operations_exit_labels,\
           entries_today, target_level_rules, stop_level_rules,\
           time_exit_rules, time_exit_loss_rules, time_exit_gain_rules,\
           money_stoploss_rules, percent_stoploss_rules, money_target_rules, percent_target_rules,\
           enter_label_list, exit_label_list,\
           mp, barssinceentry, shares_list, entry_price_list, exit_price_list,\
           open_trade_list, closed_trade_list,\
           closed_equity_list, open_equity_list,\
           adverse_excursion_list, favorable_excursion_list,\
           operations_max_adverse_excursion, operations_max_favorable_excursion, operations_duration,\
           operations_costs


def apply_trading_system(dataset, instrument, quantity, margin_percent, bigpointvalue, tick, direction, 
                         costs_fixed, costs_variable, costs_pershares,
                         order_type, enter_level, enter_rules, max_intraday_operations, 
                         exit_rules, exit_rules_loss, exit_rules_gain,
                         target_level, stop_level,
                         time_exit, time_exit_loss, time_exit_gain,
                         money_stoploss, money_target, 
                         percent_stoploss, min_money_percent_stoploss,
                         percent_target, min_money_percent_target, writelog, exit_on_entrybar, consecutive_trades):
    """
    New Numpy engine with Numba
    """
    
    import datetime
    print("")
    start = datetime.datetime.now()
    print("Elaboration starting at:", start)
    
    if money_stoploss < 0:
        money_stoploss = -money_stoploss
        
    np_dayofweek = dataset.index.dayofweek.values
    np_day = dataset.index.day.values
    np_month = dataset.index.month.values
    np_year = dataset.index.year.values
    
    np_open = dataset.open.values
    np_high = dataset.high.values
    np_low = dataset.low.values
    np_close = dataset.close.values
    np_volume = dataset.volume.values
    
    # to delete to improve speed
    if "daily_open" in dataset:
        np_openD = dataset.daily_open.values
    if "daily_open1" in dataset:
        np_openD1 = dataset.daily_open1.values
    if "daily_high1" in dataset:
        np_highD1 = dataset.daily_high1.values
    if "daily_low1" in dataset:
        np_lowD1 = dataset.daily_low1.values
    if "daily_close1" in dataset:
        np_closeD1 = dataset.daily_close1.values
    
    if isinstance(enter_rules, pd.Series) != True:
        if enter_rules == True:
            np_enter_rules = np.ones(len(np_open), dtype=bool)
        elif enter_rules == False:
            np_enter_rules = np.zeros(len(np_open), dtype=bool)
    else:
        np_enter_rules = enter_rules.values
        
    if isinstance(exit_rules, pd.Series) != True:
        if exit_rules == True:
            np_exit_rules = np.ones(len(np_open), dtype=bool)
        elif exit_rules == False:
            np_exit_rules = np.zeros(len(np_open), dtype=bool)
    else:
        np_exit_rules = exit_rules.values
        
    if isinstance(exit_rules_loss, pd.Series) != True:
        if exit_rules_loss == True:
            np_exit_rules_loss = np.ones(len(np_open), dtype=bool)
        elif exit_rules_loss == False:
            np_exit_rules_loss = np.zeros(len(np_open), dtype=bool)
    else:
        np_exit_rules_loss = exit_rules_loss.values
        
    if isinstance(exit_rules_gain, pd.Series) != True:
        if exit_rules_gain == True:
            np_exit_rules_gain = np.ones(len(np_open), dtype=bool)
        elif exit_rules_gain == False:
            np_exit_rules_gain = np.zeros(len(np_open), dtype=bool)
    else:
        np_exit_rules_gain = exit_rules_gain.values
        
    if isinstance(stop_level, pd.Series) != True:
        np_stop_level = np.zeros(len(np_open))
    else:
        np_stop_level = stop_level.values
        
    if isinstance(target_level, pd.Series) != True:
        np_target_level = np.zeros(len(np_open))
    else:
        np_target_level = target_level.values

    np_dates = dataset.index.values    
    np_enter_level = enter_level.values   
    
    operations, operations_exit_date, operations_entry_date,\
    operations_shares,operations_entry_labels,\
    operations_entry_price, operations_exit_price,operations_exit_labels,\
    entries_today, target_level_rules, stop_level_rules,\
    time_exit_rules, time_exit_loss_rules, time_exit_gain_rules,\
    money_stoploss_rules, percent_stoploss_rules, money_target_rules, percent_target_rules,\
    enter_label_list, exit_label_list,\
    mp, barssinceentry, shares_list, entry_price_list, exit_price_list,\
    open_trade_list, closed_trade_list,\
    closed_equity_list, open_equity_list,\
    adverse_excursion_list, favorable_excursion_list,\
    operations_max_adverse_excursion, operations_max_favorable_excursion, operations_duration,\
    operations_costs =\
    core(np_dayofweek, np_day, np_month, np_year,
         instrument, quantity, margin_percent, bigpointvalue, tick, direction,
         costs_fixed, costs_variable, costs_pershares, order_type,
         np_open, np_high, np_low, np_close, np_volume, np_dates,
         np_enter_level, np_enter_rules, max_intraday_operations,
         np_exit_rules, np_exit_rules_loss, np_exit_rules_gain,
         np_target_level, np_stop_level,
         time_exit, time_exit_loss, time_exit_gain,
         money_stoploss, money_target, 
         percent_stoploss, min_money_percent_stoploss,
         percent_target, min_money_percent_target, exit_on_entrybar, consecutive_trades)
        
    end = datetime.datetime.now()
    print("Elaboration completed at:", end, "in", end - start)
    print("")  
           
    if writelog == True:
        log = open('log.txt', 'w')
        log = open('log.txt', 'a')
        if "daily_open" in dataset:
            log.write("i," + "dates," + "day_of_week," +\
                      "day_of_month," + "month," + "year," +\
                      "open," + "high," + "low," +\
                      "close," + "volume," +\
                      "openD," + "openD1," + "highD1," + "lowD1," + "closeD1," +\
                      "entries_today," + "enter_level," +\
                      "enter_rules," + "exit_rules," +\
                      "exit_rules_loss," + "exit_rules_gain," +\
                      "target_level_rules," + "stop_level_rules," +\
                      "time_exit_rules," + "time_exit_loss_rules," +\
                      "time_exit_gain_rules," +\
                      "money_stoploss_rules," + "percent_stoploss_rules," +\
                      "money_target_rules," + "percent_target_rules," +\
                      "enter_label," + "exit_label," +\
                      "mp," + "barssinceentry," + "shares," +\
                      "entry_price," + "exit_price," +\
                      "open_trade," + "closed_trade," +\
                      "closed_equity," + "open_equity," +\
                      "adverse_excursion," + "favorable_excursion," + "capital \n")
            for i in range(len(np_open)):      
                log.write(str(i) + "," + str(np_dates[i]) + "," + str(np_dayofweek[i]) + "," +\
                          str(np_day[i]) + "," + str(np_month[i]) + "," + str(np_year[i]) + "," +\
                          str(np_open[i]) + "," + str(np_high[i]) + "," + str(np_low[i]) + "," +\
                          str(np_close[i]) + "," + str(np_volume[i]) + "," +\
                          str(np_openD[i]) + "," + str(np_openD1[i]) +  "," +\
                          str(np_highD1[i]) +  "," + str(np_lowD1[i]) +  "," + str(np_closeD1[i]) + "," +\
                          str(entries_today[i]) + "," + str(np_enter_level[i]) + "," +\
                          str(np_enter_rules[i]) + "," + str(np_exit_rules[i]) + "," +\
                          str(np_exit_rules_loss[i]) +  "," + str(np_exit_rules_gain[i]) + "," +\
                          str(target_level_rules[i]) + "," + str(stop_level_rules[i]) + "," +\
                          str(time_exit_rules[i]) + "," + str(time_exit_loss_rules[i]) + "," +\
                          str(time_exit_gain_rules[i]) + "," +\
                          str(money_stoploss_rules[i]) + "," + str(percent_stoploss_rules[i]) + "," +\
                          str(money_target_rules[i]) + "," + str(percent_target_rules[i]) + "," +\
                          enter_label_list[i] + "," + exit_label_list[i] + "," +\
                          str(mp[i]) + "," + str(barssinceentry[i]) + "," + str(shares_list[i]) + "," +\
                          str(entry_price_list[i]) + "," + str(exit_price_list[i]) + "," +\
                          str(open_trade_list[i]) + "," + str(closed_trade_list[i]) + "," +\
                          str(closed_equity_list[i]) + "," + str(open_equity_list[i]) + "," +\
                          str(adverse_excursion_list[i]) + "," + str(favorable_excursion_list[i]) + "," +\
                          str(shares_list[i] * entry_price_list[i] * bigpointvalue) + "\n")
        else:
            log.write("i," + "dates," + "day_of_week," +\
                      "day_of_month," + "month," + "year," +\
                      "open," + "high," + "Low," +\
                      "close," + "volume," +\
                      "entries_today," + "enter_level," +\
                      "enter_rules," + "exit_rules," +\
                      "exit_rules_loss," + "exit_rules_gain," +\
                      "target_level_rules," + "stop_level_rules," +\
                      "time_exit_rules," + "time_exit_loss_rules," +\
                      "time_exit_gain_rules," +\
                      "money_stoploss_rules," + "percent_stoploss_rules," +\
                      "money_target_rules," + "percent_target_rules," +\
                      "enter_label," + "exit_label," +\
                      "mp," + "barssinceentry," + "shares," +\
                      "entry_price," + "exit_price," +\
                      "open_trade," + "closed_trade," +\
                      "closed_equity," + "open_equity," +\
                      "adverse_excursion," + "favorable_excursion," +\
                      "capital \n")
            for i in range(len(np_open)):      
                log.write(str(i) + "," + str(np_dates[i]) + "," + str(np_dayofweek[i]) + "," +\
                          str(np_day[i]) + "," + str(np_month[i]) + "," + str(np_year[i]) + "," +\
                          str(np_open[i]) + "," + str(np_high[i]) + "," + str(np_low[i]) + "," +\
                          str(np_close[i]) + "," + str(np_volume[i]) + "," +\
                          str(entries_today[i]) + "," + str(np_enter_level[i]) + "," +\
                          str(np_enter_rules[i]) + "," + str(np_exit_rules[i]) + "," +\
                          str(np_exit_rules_loss[i]) + "," + str(np_exit_rules_gain[i]) + "," +\
                          str(target_level_rules[i]) + "," + str(stop_level_rules[i]) + "," +\
                          str(time_exit_rules[i]) + "," + str(time_exit_loss_rules[i]) + "," +\
                          str(time_exit_gain_rules[i]) + "," +\
                          str(money_stoploss_rules[i]) + "," + str(percent_stoploss_rules[i]) + "," +\
                          str(money_target_rules[i]) + "," + str(percent_target_rules[i]) + "," +\
                          enter_label_list[i] + "," + exit_label_list[i] + "," +\
                          str(mp[i]) + "," + str(barssinceentry[i]) + "," + str(shares_list[i]) + "," +\
                          str(entry_price_list[i]) + "," + str(exit_price_list[i]) + "," +\
                          str(open_trade_list[i]) + "," + str(closed_trade_list[i]) + "," +\
                          str(closed_equity_list[i]) + "," + str(open_equity_list[i]) + "," +\
                          str(adverse_excursion_list[i]) + "," + str(favorable_excursion_list[i]) + "," +\
                          str(shares_list[i] * entry_price_list[i] * bigpointvalue) + "\n")  
        log.close()
        print("log.txt saved...\n")
        
    a1 = len(operations_entry_date)
    a2 = len(operations_exit_date)
    print("Consistency check: entries:", a1, "exits:", a2)
    print("")
    if a1 != a2:
        print("ENGINE ERROR: entries number different from exits number!")
        print("")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if len(operations_entry_date) != 0 and len(operations_exit_date) != 0:
        if "last" in exit_label_list[-1]:
            print("Last trade still open: we close it on last bar and we compute open position as closed position!")
            print("")
        
    if len(operations_entry_date) != 0 and len(operations_exit_date) != 0:
        
        tradelist = pd.DataFrame(index = operations_exit_date)
        tradelist["id"] = pd.DataFrame(np.arange(len(operations_entry_date)) + 1, index = operations_exit_date)
        tradelist["entry_date"] = operations_entry_date
        tradelist["entry_label"] = operations_entry_labels
        tradelist["quantity"] = operations_shares
        tradelist["entry_price"] = operations_entry_price
        tradelist["exit_date"] = operations_exit_date
        tradelist["exit_label"] = operations_exit_labels
        tradelist["exit_price"] = operations_exit_price
        tradelist["bars_in_trade"] = operations_duration
        tradelist["mae"] = operations_max_adverse_excursion
        tradelist["mfe"] = operations_max_favorable_excursion
        tradelist["operations"] = operations
        tradelist["capital"] = tradelist["quantity"] * tradelist["entry_price"] * bigpointvalue
        tradelist["costs"] = operations_costs

        equities = pd.DataFrame(open_equity_list, index = dataset.index)
        equities.columns = ["open_equity"]
        equities["closed_equity"] = closed_equity_list

        operation_equity = tradelist.operations.cumsum()
        
        return tradelist, equities.open_equity, equities.closed_equity, operation_equity
        
    else:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    

def apply_trading_system_opt(dataset, instrument, quantity, margin_percent, bigpointvalue, tick, direction, 
                             costs_fixed, costs_variable, costs_pershares,
                             order_type, enter_level, enter_rules, max_intraday_operations, 
                             exit_rules, exit_rules_loss, exit_rules_gain,
                             target_level, stop_level,
                             time_exit, time_exit_loss, time_exit_gain,
                             money_stoploss, money_target, 
                             percent_stoploss, min_money_percent_stoploss,
                             percent_target, min_money_percent_target, writelog, exit_on_entrybar, consecutive_trades):
    """
    New Numpy engine with Numba for Optimization
    """
    
    import datetime
    #print("")
    start = datetime.datetime.now()
    #print("Elaboration starting at:", start)
    
    if money_stoploss < 0:
        money_stoploss = -money_stoploss
        
    np_dayofweek = dataset.index.dayofweek.values
    np_day = dataset.index.day.values
    np_month = dataset.index.month.values
    np_year = dataset.index.year.values
    
    np_open = dataset.open.values
    np_high = dataset.high.values
    np_low = dataset.low.values
    np_close = dataset.close.values
    np_volume = dataset.volume.values
    
    # to delete to improve speed
    if "daily_open" in dataset:
        np_openD = dataset.daily_open.values
    if "daily_open1" in dataset:
        np_openD1 = dataset.daily_open1.values
    if "daily_high1" in dataset:
        np_highD1 = dataset.daily_high1.values
    if "daily_low1" in dataset:
        np_lowD1 = dataset.daily_low1.values
    if "daily_close1" in dataset:
        np_closeD1 = dataset.daily_close1.values
    
    if isinstance(enter_rules, pd.Series) != True:
        if enter_rules == True:
            np_enter_rules = np.ones(len(np_open), dtype=bool)
        elif enter_rules == False:
            np_enter_rules = np.zeros(len(np_open), dtype=bool)
    else:
        np_enter_rules = enter_rules.values
        
    if isinstance(exit_rules, pd.Series) != True:
        if exit_rules == True:
            np_exit_rules = np.ones(len(np_open), dtype=bool)
        elif exit_rules == False:
            np_exit_rules = np.zeros(len(np_open), dtype=bool)
    else:
        np_exit_rules = exit_rules.values
        
    if isinstance(exit_rules_loss, pd.Series) != True:
        if exit_rules_loss == True:
            np_exit_rules_loss = np.ones(len(np_open), dtype=bool)
        elif exit_rules_loss == False:
            np_exit_rules_loss = np.zeros(len(np_open), dtype=bool)
    else:
        np_exit_rules_loss = exit_rules_loss.values
        
    if isinstance(exit_rules_gain, pd.Series) != True:
        if exit_rules_gain == True:
            np_exit_rules_gain = np.ones(len(np_open), dtype=bool)
        elif exit_rules_gain == False:
            np_exit_rules_gain = np.zeros(len(np_open), dtype=bool)
    else:
        np_exit_rules_gain = exit_rules_gain.values
        
    if isinstance(stop_level, pd.Series) != True:
        np_stop_level = np.zeros(len(np_open))
    else:
        np_stop_level = stop_level.values
        
    if isinstance(target_level, pd.Series) != True:
        np_target_level = np.zeros(len(np_open))
    else:
        np_target_level = target_level.values

    np_dates = dataset.index.values    
    np_enter_level = enter_level.values   
    
    operations, operations_exit_date, operations_entry_date,\
    operations_shares,operations_entry_labels,\
    operations_entry_price, operations_exit_price,operations_exit_labels,\
    entries_today, target_level_rules, stop_level_rules,\
    time_exit_rules, time_exit_loss_rules, time_exit_gain_rules,\
    money_stoploss_rules, percent_stoploss_rules, money_target_rules, percent_target_rules,\
    enter_label_list, exit_label_list,\
    mp, barssinceentry, shares_list, entry_price_list, exit_price_list,\
    open_trade_list, closed_trade_list,\
    closed_equity_list, open_equity_list,\
    adverse_excursion_list, favorable_excursion_list,\
    operations_max_adverse_excursion, operations_max_favorable_excursion, operations_duration,\
    operations_costs =\
    core(np_dayofweek, np_day, np_month, np_year,
         instrument, quantity, margin_percent, bigpointvalue, tick, direction,
         costs_fixed, costs_variable, costs_pershares, order_type,
         np_open, np_high, np_low, np_close, np_volume, np_dates,
         np_enter_level, np_enter_rules, max_intraday_operations,
         np_exit_rules, np_exit_rules_loss, np_exit_rules_gain,
         np_target_level, np_stop_level,
         time_exit, time_exit_loss, time_exit_gain,
         money_stoploss, money_target, 
         percent_stoploss, min_money_percent_stoploss,
         percent_target, min_money_percent_target, exit_on_entrybar, consecutive_trades)
        
    end = datetime.datetime.now()
    #print("Elaboration completed at:", end, "in", end - start)
    #print("")  
        
    a1 = len(operations_entry_date)
    a2 = len(operations_exit_date)
    #print("Consistency check: entries:", a1, "exits:", a2)
    #print("")
    if a1 != a2:
        print("ENGINE ERROR: entries number different from exits number!")
        print("")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    """
    if len(operations_entry_date) != 0 and len(operations_exit_date) != 0:
        if "last" in exit_label_list[-1]:
            print("Last trade still open: we close it on last bar and we compute open position as closed position!")
            print("")
    """
        
    if len(operations_entry_date) != 0 and len(operations_exit_date) != 0:
        
        tradelist = pd.DataFrame(index = operations_exit_date)
        tradelist["id"] = pd.DataFrame(np.arange(len(operations_entry_date)) + 1, index = operations_exit_date)
        tradelist["entry_date"] = operations_entry_date
        tradelist["entry_label"] = operations_entry_labels
        tradelist["quantity"] = operations_shares
        tradelist["entry_price"] = operations_entry_price
        tradelist["exit_date"] = operations_exit_date
        tradelist["exit_label"] = operations_exit_labels
        tradelist["exit_price"] = operations_exit_price
        tradelist["bars_in_trade"] = operations_duration
        tradelist["mae"] = operations_max_adverse_excursion
        tradelist["mfe"] = operations_max_favorable_excursion
        tradelist["operations"] = operations
        tradelist["capital"] = tradelist["quantity"] * tradelist["entry_price"] * bigpointvalue
        tradelist["costs"] = operations_costs
        
        equities = pd.DataFrame(open_equity_list, index = dataset.index)
        equities.columns = ["open_equity"]
        equities["closed_equity"] = closed_equity_list

        operation_equity = tradelist.operations.cumsum()
        
        return tradelist, equities.open_equity, equities.closed_equity, operation_equity
        
    else:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()