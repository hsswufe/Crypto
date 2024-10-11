
import logging
import time
import ccxt
import pandas as pd
import schedule
from tqdm import tqdm
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
target_pool_logger = setup_logger('target_pool_logger', r'target_pool_logger.txt')



okex = ccxt.okx({
    'apiKey':"yours",
    'secret':"yours",
'password': 'yours',   # 更新为API的密码
}) # 这里我起名为okex， 也就是文档中的exchange

okex.proxies = {
'http':'proxy.bigquant.ai:31701',
'https':'proxy.bigquant.ai:31701',
'all':'socks5://proxy.bigquant.ai:31710'
}
# okex.proxies = {
#     'http': 'http://127.0.0.1:7890',
#     'https': 'http://127.0.0.1:7890',
#     'all': 'socks5://127.0.0.1:7890'
# }
okex.set_sandbox_mode(False)


class swap_pool:
    def __init__(self,funding_threshold= 0,simulation=False, ):
        '''

        :param funding_threshold:  资金费率的阈值，本策略是空期货收取资金费，因此选择资金费率为正的标的进行交易
        :param simulation:  是否开启模拟交易
        '''
        self.exchange = okex
        self.funding_threshold = funding_threshold
        # 先做一个行情数据的处理问题
        self.market = pd.DataFrame(self.exchange.fetch_markets())[['id', 'symbol', 'type', 'info']]
        # 获取币种的代码
        self.spot_df = self.market[self.market['type'] == 'spot']
        # 获取永续期货的代码
        self.swap_df = self.market[self.market['type'] == 'swap']
        # 匹配这个期货合约和现货的代码
        # 可交易期货的现货代码
        self.spot_symbol = [code for code in [id[:-5] for id in self.swap_df.id.values if id[:-5] in self.spot_df.id.values]]
        self.swap_symbol = [str(id) + '-SWAP' for id in self.spot_symbol]

    def _get_fundingrate_list(self, ):
        coin_rate = []
        # 总进度
        total = len(self.swap_symbol)
        with tqdm(total=total) as pbar:
            pbar.set_description('get_fundingrate Processing:')
            for i, coin in enumerate(self.swap_symbol):
                _finding_rate = float(self.exchange.fetch_funding_rate(symbol=coin)['info']['fundingRate'])
                coin_rate.append((coin, _finding_rate))
                pbar.update(1)
            pbar.close()
        fundingrate_df = pd.DataFrame(coin_rate, columns=['symbol', 'fundingRate'])
        del _finding_rate, coin_rate
        return fundingrate_df[fundingrate_df['fundingRate'] > self.funding_threshold]

    def fetch_ohlcv_data(self, symbol, timeframe='1m', limit=30):
        data = pd.DataFrame(okex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit),
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
        data['symbol'] = symbol
        return data

    def _factor_cal(self,rank = 50):
        symbols = self._get_fundingrate_list().symbol.tolist()
        total = len(symbols)
        # 初始化一个空的dataframe
        swap_factor = []
        with tqdm(total=total) as pbar:
            pbar.set_description('Processing:')
            for i, coin in enumerate(symbols):
                price =  self.fetch_ohlcv_data(coin, timeframe='1h', limit=24)
                # 计算收益率
                price['ret'] = price['close'].pct_change()
                price.dropna(inplace = True)
                # 滚动计算收益率过去24小时的波动率
                price['ret_std'] = price['ret'].std()
                # 滚动计算过去24小时的成交量
                price['vol_sum'] = price['volume'].sum()
                price.dropna(inplace=True)
                swap_factor.append({'coin':coin, 'std':price['ret'].std(), 'vol':price['volume'].sum()})
                pbar.update(1)
        swap_factor = pd.DataFrame(swap_factor)
        # 依据成交量和波动率进行打分
        # 这里的思路是对两个因子进行排名之后相加作为最终的得分结果（取得分最小的前5个）
        swap_factor['ret_std_rank'] = swap_factor['std'].rank(method='min')
        swap_factor['vol_sum_rank'] = swap_factor['vol'].rank(method='min')
        swap_factor['score'] = swap_factor['ret_std_rank'] + swap_factor['vol_sum_rank']
        return swap_factor.nlargest(rank, 'score').coin.tolist()

def main():
    swappool = swap_pool()
    list = swappool._factor_cal()
    target_pool_logger.info(f'The filtered currency pairs are :{list}')




if __name__ == '__main__':
    # schedule.every().day.at("00:01").do(main)
    # schedule.every().day.at("08:01").do(main)
    # schedule.every().day.at("16:01").do(main)
    # while True:
    #     schedule.run_pending()
    #     time.sleep(50)
    main()



