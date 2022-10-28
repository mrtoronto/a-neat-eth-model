import logging
import numpy as np

FEE = 0.999

def _buy_coin_for_usd(balances, price, n_usd, coin):
    # Check if you have enough usd to buy
    ### https://www.binance.com/en/support/faq/115000429332
    if balances['usd'] > n_usd:
        ### if so, add eth to wallet
        ### Take fee from input USD
        balances[coin] = balances[coin] + ((n_usd * FEE) / price)
    # remove USD (happens even if agent is too poor which acts like punishment for wrong decision)
    balances['usd'] = max(0, balances['usd'] - n_usd)
    return balances

def _sell_n_coin(balances, price, n_coin, coin):
    if balances[coin] > (n_coin):
        balances['usd'] = balances['usd'] + ((n_coin * FEE) * price)
    balances[coin] = max(0, balances[coin] - (n_coin))
    return balances



def _process_action_buy_percInts_v2(action, balances, open_price):
    action_val = np.argmax(action)
    # 5% balance buy
    if action_val == 0:
        balances = _buy_coin_for_usd(balances, open_price, balances['usd'] * 0.1, coin='eth')
    elif action_val == 1:
        balances = _buy_coin_for_usd(balances, open_price, balances['usd'] * 0.05, coin='eth')
    # 3% balance buy
    elif action_val == 2:
        balances = _buy_coin_for_usd(balances, open_price, balances['usd'] * 0.03, coin='eth')
    # 1% balance buy
    elif action_val == 3:
        balances = _buy_coin_for_usd(balances, open_price, balances['usd'] * 0.01, coin='eth')
    # 0% buy
    elif action_val == 4:
        return balances
    # 1% sell
    ### Try with _sell_n_eth instead. More logical and doesn't back model into a corner with low money
    elif action_val == 5:
        balances = _sell_n_coin(balances, open_price, balances['eth'] * 0.01, coin='eth')
    # 3% sell
    elif action_val == 6:
        balances = _sell_n_coin(balances, open_price, balances['eth'] * 0.03, coin='eth')
    # 5% sell
    elif action_val == 7:
        balances = _sell_n_coin(balances, open_price, balances['eth'] * 0.05, coin='eth')
    elif action_val == 8:
        balances = _sell_n_coin(balances, open_price, balances['eth'] * 0.1, coin='eth')
    else:
        logging.info(f'WE HAVE AN ISSUE IN _process_action_buy_percInts: {action} | {action_val}')
        return balances
    return balances

def _process_action_buy_percInts_v3(action, balances, open_price):
    action_val = np.argmax(action)
    # 5% balance buy
    if action_val == 0:
        balances = _buy_coin_for_usd(balances, open_price, balances['usd'] * 0.25, coin='eth')
    elif action_val == 1:
        balances = _buy_coin_for_usd(balances, open_price, balances['usd'] * 0.1, coin='eth')
    # 3% balance buy
    elif action_val == 2:
        balances = _buy_coin_for_usd(balances, open_price, balances['usd'] * 0.05, coin='eth')
    # 1% balance buy
    elif action_val == 3:
        balances = _buy_coin_for_usd(balances, open_price, balances['usd'] * 0.03, coin='eth')
    # 0% buy
    elif action_val == 4:
        return balances
    # 1% sell
    ### Try with _sell_n_eth instead. More logical and doesn't back model into a corner with low money
    elif action_val == 5:
        balances = _sell_n_coin(balances, open_price, balances['eth'] * 0.03, coin='eth')
    # 3% sell
    elif action_val == 6:
        balances = _sell_n_coin(balances, open_price, balances['eth'] * 0.05, coin='eth')
    # 5% sell
    elif action_val == 7:
        balances = _sell_n_coin(balances, open_price, balances['eth'] * 0.1, coin='eth')
    elif action_val == 8:
        balances = _sell_n_coin(balances, open_price, balances['eth'] * 0.25, coin='eth')
    else:
        logging.info(f'WE HAVE AN ISSUE IN _process_action_buy_percInts: {action} | {action_val}')
        return balances
    return balances

def _process_action_eth_btc(action, balances, eth_price, btc_price):
    action_val = np.argmax(action)
    # 1% balance buy
    if action_val == 0:
        balances = _buy_coin_for_usd(balances, eth_price, balances['usd'] * 0.20, coin='eth')
    elif action_val == 1:
        balances = _buy_coin_for_usd(balances, btc_price, balances['usd'] * 0.20, coin='btc')

    elif action_val == 2:
        balances = _buy_coin_for_usd(balances, eth_price, balances['usd'] * 0.1, coin='eth')
    elif action_val == 3:
        balances = _buy_coin_for_usd(balances, btc_price, balances['usd'] * 0.1, coin='btc')

    elif action_val == 4:
        balances = _buy_coin_for_usd(balances, eth_price, balances['usd'] * 0.05, coin='eth')
    elif action_val == 5:
        balances = _buy_coin_for_usd(balances, btc_price, balances['usd'] * 0.05, coin='btc')

    elif action_val == 6:
        balances = _buy_coin_for_usd(balances, eth_price, balances['usd'] * 0.025, coin='eth')
    elif action_val == 7:
        balances = _buy_coin_for_usd(balances, btc_price, balances['usd'] * 0.025, coin='btc')

    elif action_val == 8:
        return balances

    elif action_val == 9:
        balances = _sell_n_coin(balances, eth_price, balances['eth'] * 0.025, coin='eth')
    
    elif action_val == 10:
        balances = _sell_n_coin(balances, btc_price, balances['btc'] * 0.025, coin='btc')

    elif action_val == 11:
        balances = _sell_n_coin(balances, eth_price, balances['eth'] * 0.05, coin='eth')
    
    elif action_val == 12:
        balances = _sell_n_coin(balances, btc_price, balances['btc'] * 0.05, coin='btc')

    elif action_val == 13:
        balances = _sell_n_coin(balances, eth_price, balances['eth'] * 0.1, coin='eth')
    
    elif action_val == 14:
        balances = _sell_n_coin(balances, btc_price, balances['btc'] * 0.1, coin='btc')

    elif action_val == 15:
        balances = _sell_n_coin(balances, eth_price, balances['eth'] * 0.2, coin='eth')
    
    elif action_val == 16:
        balances = _sell_n_coin(balances, btc_price, balances['btc'] * 0.2, coin='btc')
    
    else:
        logging.info(f'WE HAVE AN ISSUE IN _process_action_buy_percInts: {action} | {action_val}')
        return balances
    return balances