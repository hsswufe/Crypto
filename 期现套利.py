import ast
import asyncio
import logging
import re
import threading
import time

import ccxt
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
# 定义一个类

# 设置不同的日志记录器
def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


# 创建不同的日志记录器
balance_logger = setup_logger('balance_logger', 'balance_log.txt')
openpos_logger = setup_logger('openpos_logger', 'openpos_log.txt')
closepos_logger = setup_logger('closepos_logger', 'closepos_log.txt')



okex = ccxt.okx({
    'apiKey':"yours",
    'secret':"yours",
'password': 'yours',   # 更新为API的密码
}) # 这里我起名为okex， 也就是文档中的exchange


okex.set_sandbox_mode(False)


class swap_spot_pair:
    def __init__(self,simulation=False, ):
        '''
        :param simulation:  是否开启模拟交易
        '''
        self.exchange = okex

    def fetch_ohlcv_data(self, symbol, timeframe='1m', limit=180):
        data = pd.DataFrame(okex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit),
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
        data['symbol'] = symbol
        return data


    def trade_pool(self):
        """
        交易池
        :return: 交易池
        """
        # 从日志中获取最新的币对池
        df = pd.read_csv('target_pool_logger.txt', header=None, sep='\t')
        # 使用正则表达式匹配列表部分
        match = re.search(r"\[(.*?)\]", df.iloc[-1, :].values[0])
        # 使用ast.literal_eval安全地将字符串转换为列表
        symbols = ast.literal_eval(match.group(0))
        # 获取合约的最新价格
        trade_list = []
        for coin in symbols:
            basis = okex.fetch_ticker(coin[:-5])['last'] - okex.fetch_ticker(coin)['last']
            basis_data = pd.merge(
                self.fetch_ohlcv_data(coin),
                self.fetch_ohlcv_data(coin[:-5]),
                how='left', on='date', suffixes=('_swap', '_spot')
            )
            basis_ago_percentile_20 = np.percentile(basis_data['close_spot'] - basis_data['close_swap'], 20)
            if basis < basis_ago_percentile_20:
                trade_list.append(coin)
        return trade_list

def pos_open(exchange, symbols, pos_num, leverage, order_price):
    '''
    下单逻辑是等权重持仓
    :param exchange:
    :param symbols:  下单标的池
    :param order_price: 下单的总钱数（如果不指定，那就是账户余额的1/10）
    :param pos_num:  目标持仓数
    :param leverage: 杠杆倍数
    :return:
    '''
    # 为各个品种设置杠杆为1
    for symbol in symbols:
        exchange.set_leverage(symbol=symbol, leverage=leverage)
    for symbol in symbols:
        # 设置最大下单尝试次数，超出次数停止对该品种下单
        pos_df = pd.DataFrame(exchange.fetch_positions())
        if pos_df.empty:
            _symbols = []
        else:
            _symbols = pos_df[pos_df['info'].apply(lambda x: x['instType'] == 'SWAP')]['info'].apply(
                lambda x: x['instId']).tolist()
        if  symbol in _symbols:
            continue
        else:
            max_attempts = 5
            attempt = 0
            params = {"tdMode": "cross", "posSide": "short"}
            try:
                _swap_bid = okex.fetch_ticker(symbol=symbol)['bid']# 获取最新买一价
            except Exception as e:
                print(f"Error fetching {symbol} data : {e}", end='')
                continue
            contractSize = exchange.market(symbol=symbol)['contractSize']  # 单张合约对应的数字货币的数量
            presicion_swap = 1 / exchange.market(symbol=symbol)['precision']['amount']  # 保留小数位数
            swap_amount = (((order_price / min(len(symbols), pos_num)) / 2) / _swap_bid) / contractSize  # 下单的张数
            swap_amount = max((int(swap_amount * presicion_swap) / presicion_swap), float(exchange.market(symbol=symbol)['info']['minSz']), float(okex.market(symbol[:-5])['info']['minSz'])/float(okex.market(symbol)['contractSize']))# 保证最小下单数量
            print(f'===================>{symbol} 开始下单, 下单数量为{swap_amount}')
            while attempt < max_attempts:
                swap_order_id = exchange.create_order(symbol=symbol, side='sell', type='limit', price=_swap_bid, params=params,
                                                    amount=swap_amount)['id']
                openpos_logger.info(
                    f'{symbol} short {swap_order_id} has send ,num is {swap_amount}'
                )
                time.sleep(3)
                swap_order_status = exchange.fetch_order(swap_order_id, symbol)['status']

                if swap_order_status == 'closed':
                    print(f'===================>{symbol} 空单已经完成,数量为 {swap_amount}')
                    openpos_logger.info(
                        f'{symbol} short {swap_order_id} has filled,num is {swap_amount}'
                    )
                    spot_amount = swap_amount*contractSize*1.02  # 保证不会随着时间推移产生不满足最小平仓数量的问题
                    # 保证市价单可以正常下单
                    print(f'===================>{symbol[:-5]} 开始下单, 下单数量为{spot_amount}')
                    spot_order_id = \
                    exchange.create_order(symbol[:-5], side='buy', type='market', params={}, amount=spot_amount)['id']
                    while True:
                        spot_order_status = exchange.fetch_order(spot_order_id, symbol[:-5])['status']
                        if spot_order_status == 'closed':
                            print(f'===================>{symbol[:-5]} 多单已经完成,数量为{spot_amount}')
                            openpos_logger.info(
                                f'market {symbol[:-5]} spot buy {spot_order_id} has filled,num is {spot_amount}'
                            )
                            break
                    break
                else:
                    openpos_logger.info(
                        f'swap  short {swap_order_id} not filled, Attempt to cancel the order and place a new one'
                    )
                    exchange.cancel_order(swap_order_id, symbol)
                    attempt += 1
                    if attempt == max_attempts - 1:
                        openpos_logger.info(
                            f'The number of attempts has been exhausted, unable to place an order{symbol}'
                        )
        openpos_logger.info('=' * 50)

def closepos(exchange):
    '''
    这个平仓机制要求开始时的账户内最好没有任何持仓
    还需要考虑平仓时存在的单边持仓的问题
    :return:
    '''
    def fetch_ohlcv_data_24h( symbol, timeframe='1h', limit=24):
        data = pd.DataFrame(okex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit),
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
        data['symbol'] = symbol
        return data

    def fetch_ohlcv_data_3h( symbol, timeframe='1m', limit=180):
        data = pd.DataFrame(okex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit),
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
        data['symbol'] = symbol
        return data

    # 强平机制
    closepos_logger.info('start closepos')
    pos_df = pd.DataFrame(exchange.fetch_positions())  # 获取持仓
    if len(pos_df) == 0:
        closepos_logger.info('position is None')
        return
    # 下面进行一个简单的筛选,判定那些交易对在平仓时已经持仓了,即以: 合约(空仓)-现货(多仓)的形式存在
    _symbols = pos_df[pos_df['info'].apply(lambda x: x['instType'] == 'SWAP')]['info'].apply(
        lambda x: x['instId']).tolist()
    symbols = []
    for symbol in _symbols:
        match = re.search(r'^(.*?)-USDT-SWAP', symbol)
        part = match.group(1)
        if part in list(okex.fetch_balance().keys())[2:-5]:
            symbols.append(symbol)
    print(f'===================>满足期现同时持仓的币对数为{len(symbols)}')

    for symbol in symbols:
        basis = okex.fetch_ticker(symbol[:-5])['last'] - okex.fetch_ticker(symbol)['last']
        basis_data_24h = pd.merge(
            fetch_ohlcv_data_24h(symbol),
            fetch_ohlcv_data_24h(symbol[:-5]),
            how='left', on='date', suffixes=('_swap', '_spot')
        )
        basis_data_3h = pd.merge(
            fetch_ohlcv_data_3h(symbol),
            fetch_ohlcv_data_3h(symbol[:-5]),
            how='left', on='date', suffixes=('_swap', '_spot')
        )
        basis_ago_70_24h = np.percentile(basis_data_24h['close_spot'] - basis_data_24h['close_swap'], 70)
        basis_ago_70_3h = np.percentile(basis_data_3h['close_spot'] - basis_data_3h['close_swap'], 70)
        closed_thresold = 0.7*basis_ago_70_3h+0.3*basis_ago_70_24h
        print(f'===================>{symbol}基差为{basis}, 平仓阈值为{closed_thresold}')
        if basis > closed_thresold:
            print(f'===================>{symbol} 满足平仓条件')
            code = exchange.market(symbol)['base']
            position_swap = float(pos_df[pos_df['symbol'].str.contains(code)]['contracts'].values)  # 获取仓位内的合约数量
            params = {"tdMode": "cross", "posSide": "short", }
            swap_order = exchange.create_order(symbol, side='buy', type='market', params=params,
                                            amount=position_swap)['id']  # 平合约空头
            while True:
                time.sleep(0.1)
                swap_order_status = exchange.fetch_order(swap_order, symbol)['status']
                if swap_order_status == 'closed':
                    print(f'===================>{symbol}平仓')
                    break

            precision = 1 / float(okex.market(symbol[:-5])['precision']['amount'])
            position_spot = okex.fetch_balance()[symbol[:-10]]['free']
            position_spot = int(position_spot*precision)/precision
            spot_order = exchange.create_order(symbol[:-5], side='sell', type='market',
                                            amount=position_spot)['id']  # 平现货多头
            while True:
                time.sleep(0.1)
                spot_order_status = exchange.fetch_order(spot_order, symbol[:-5])['status']
                if spot_order_status == 'closed':
                    print(f'===================>{symbol[:-5]}平仓')
                    break
            closepos_logger.info(
                f'strong level {symbol} short {position_swap}, {symbol[:-5]} long {position_spot}')
            closepos_logger.info('=' * 50)
        else:
            print(f'===================>{symbol} 不满足平仓条件')


def singleside():
    '''
    检查是否存在单边持仓的情况,并对其进行处理,主要是针对现货仓位不断累计的问题进行修复
    :return:
    '''
    # 获取单边持仓的币种
    pos_df = pd.DataFrame(okex.fetch_positions())
    coin = list(okex.fetch_balance().keys())[2:-5]
    symbols = pos_df[pos_df['info'].apply(lambda x: x['instType'] == 'SWAP')]['info'].apply(
        lambda x: x['instId']).tolist()
    singleswap = []
    for symbol in symbols:
        match = re.search(r'^(.*?)-USDT-SWAP', symbol)
        part = match.group(1)
        if part not in coin:
            singleswap.append(symbol) # 现货仓位中不存在而合约仓位中存在的币种

    for symbol in singleswap:
        code = okex.market(symbol)['base']
        position_swap = float(pos_df[pos_df['symbol'].str.contains(code)]['contracts'].values)  # 获取仓位内的合约数量
        params = {"tdMode": "cross", "posSide": "short", }
        id = okex.create_order(symbol, side='buy', type='market', params=params,
                                        amount=position_swap)['id']  # 平合约空头
        while True:
            time.sleep(0.1)
            order_status = okex.fetch_order(id, symbol)['status']
            if order_status == 'closed':
                print(f'===================>{symbol}合约清仓')


def main():

    print(f'===================>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}开始期现套利:')
    # 获取可交易的交易对
    swap_pair = swap_spot_pair()
    symbols = swap_pair.trade_pool()
    'TURBO-USDT-SWAP' in symbols and symbols.remove('TURBO-USDT-SWAP') # 去除TURBO交易对(资金量占有过大)
    if len(symbols) == 0:
        print('===================>没有可交易的交易对')
        return
    else:
        print(f'===================>可交易的交易对为{symbols}')
        order_price =  okex.fetch_balance()['USDT']['free'] * 0.05  # 下单总金额
        free_usdt = okex.fetch_balance()['USDT']['free']
        position_num= len(pd.DataFrame(okex.fetch_positions()))
        if  free_usdt > 5 and position_num < 8:
            pos_num = len(symbols)
            pos_open(exchange=okex, symbols=symbols, order_price=order_price, pos_num=pos_num, leverage=1.5)
        else:
            print('===================>仓位已满不开仓')
            return

def log_account(exchange):
    account_value_initate = 103.6
    # 现货仓位的总价值
    spot = pd.DataFrame(index=exchange.fetch_balance()['free'].keys(), data=exchange.fetch_balance()['free'].values(),
                        columns=['free'])
    spot_values = 0
    for symbol in spot.index:
        if symbol not in ['USDT']:
            spot_values += spot[spot.index == symbol].values * exchange.fetch_ticker(symbol + '-USDT')['last']
    account_value_now = spot_values + exchange.fetch_balance()['total']['USDT']  # 当前账户总价值
    balance_logger.info(f'Account balance: {account_value_now}')
    return 0




if __name__ == '__main__':
    def run_schedule():
        while True:
            try:
                main()
                time.sleep(90)  # 每秒检查一次是否有任务需要执行
            except Exception as e:
                print(f'发生异常: {e},3s后重起运行策略')
                time.sleep(3)

    def log():
        while True:
            try:
                # schedule.run_pending()
                time.sleep(600)  # 每 10 分钟记录一次
                log_account(exchange=okex)
            except Exception as e:
                print(f'记录异常: {e}')

    # 启动 schedule 线程
    schedule_thread = threading.Thread(target=run_schedule)
    schedule_thread.start()
    # 记录线程
    log_thread = threading.Thread(target=log)
    log_thread.start()

    # 主线程继续执行其他任务
    while True:
        try:
            time.sleep(30)  # 主线程每5s醒来一次，但不会阻止 schedule 的执行
            closepos(exchange=okex)
        except Exception as e:
            print(f'发生异常: {e}')




