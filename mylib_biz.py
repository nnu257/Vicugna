from datetime import datetime, timedelta
import jpholiday


def isBizDay(day:datetime) -> bool:
    # 営業日かを判定する関数
    if day.weekday() >= 5 or jpholiday.is_holiday(day):
        return False
    else:
        return True

def day_n_far_biz(date1:datetime, date2:datetime, n:int) -> bool:
    # 二つの日付に対して、n営業日以上離れているか確認する関数

    # 離れている日にちを計算して返す
    count = 0
    # n営業日になるまで計算
    while count < n:
        count += 1
        
        date1 += timedelta(days=1)
        while not isBizDay(date1):
            date1 += timedelta(days=1)

        # その途中でdate2を追い越したらアウト
        if date1 > date2:
            return False
    return True

def afterNbizday_date(date1:datetime, n:int) -> datetime:
    # date1からn営業日後の日付を返す
    count = 0
    # n営業日になるまで計算
    while count < n:
        count += 1
        
        date1 += timedelta(days=1)
        while not isBizDay(date1):
            date1 += timedelta(days=1)
        
    return date1