# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

df.head()


# Görev 2: Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.

df.value_counts("sex")

# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
df.columns
df.nunique()

# Görev 4: pclass değişkeninin unique değerlerinin sayısını bulunuz.

df["pclass"].nunique()

# Görev 5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.

df[["pclass", "parch"]].nunique()


# Görev 6: embarked değişkeninin tipini kontrol ediniz.
# Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.

df.dtypes
df["embarked"].dtypes
df["embarked"] = df["embarked"].astype("category")


# Görev 7: embarked değeri C olanların tüm bilgilerini gösteriniz.

df[df["embarked"] == "C"].head()

# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.

df[df["embarked"] != "S"].head()

# Görev9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.

df[(df["age"] < 30) & (df["sex"] == "female")].head()

# Görev 10: Fare'i 500'den büyük veya yaşı 70’den büyük yolcuların bilgilerini gösteriniz.

df[(df["fare"] > 500) | (df["age"] > 70)].head()

# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.

df.isnull().sum()

# Görev 12: who değişkenini dataframe’den çıkarınız.
df.head()
df.drop("who", axis=1, inplace=True)

# Görev 13: deck değiskenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.

df["deck"].isnull().sum() # 688
df["deck"].mode() # C
df.value_counts("deck")["C"] # C den 59 tane var
688+59 # 747
df["deck"].fillna("C", axis=0, inplace=True)

# Görev 14: age değiskenindeki boş değerleri age değişkenin medyanı ile doldurunuz.
type(df["age"].median())
df["age"].median()
df["age"].isnull().sum()
df.value_counts("age")[28.0]
df["age"].fillna(df["age"].median(), inplace=True)


# Görev 15: survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.

df.groupby(["pclass", "sex"]).agg({
    "survived": ["sum", "count", "mean"]})

df.pivot_table("survived", "pclass", "sex", aggfunc=["sum", "count", "mean"])
# aggfunc={} ile de oluyor biraz daha farkli gosteriyor.

# Görev 16: 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 verecek bir fonksiyon yazın.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)

df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0).head()
df.head()

# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.

df = sns.load_dataset("tips")
df.head()

# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerinin sum, min, max ve mean değerlerini bulunuz.
df.head()
df["time"].value_counts()
df["time"].value_counts().min()
df["time"].value_counts().max()
df["time"].value_counts().mean()

df.groupby(["time"]).agg({
    "total_bill": ["sum", "min", "max", "mean"]})



# Görev 19: Day ve time’a göre total_bill değerlerinin sum, min, max ve mean değerlerini bulunuz.

df.head()

df.groupby(["day", "time"]).agg({
    "total_bill": ["sum", "min", "max", "mean"]})
type(df)
#
# Görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre sum, min, max ve mean değerlerini bulunuz.

df.head()
df_lunch_fem = df[(df["time"] == "Lunch") & (df["sex"] == "Female")]
type(df_lunch_fem)
df_lunch_fem.head()
df_lunch_fem.groupby("day").agg({
    "total_bill": ["sum", "min", "max", "mean"],
    "tip": ["sum", "min", "max", "mean"]})

df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").agg({
                                        "total_bill": ["sum", "min", "max", "mean"],
                                        "tip": ["sum", "min", "max", "mean"]})


# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)

df[(df["size"] < 3) & (df["total_bill"] > 10)].head()

df.loc[(df["size"] < 3) & (df["total_bill"] > 10), :].mean()
df.head()
df.loc[(df["size"] < 3) & (df["total_bill"] > 10), "total_bill"].mean()

# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturunuz.
# Her bir müşterinin ödediği totalbill ve tip in toplamını versin.

df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()

# Görev 23: Total_bill değişkeninin kadın ve erkek için ayrı ayrı ortalamasını bulunuz.
# Bulduğunuz ortalamaların altında olanlara 0, üstünde ve eşit olanlara 1 verildiği yeni bir total_bill_flag değişkeni oluşturunuz.
# Kadınlar için Female olanlarının ortalamaları, erkekler için ise Male olanların ortalamaları dikkate alınacktır.
# Parametre olarak cinsiyet ve total_bill alan bir fonksiyon yazarak başlayınız. (If-else koşulları içerecek)

total_bill_fem = df.loc[df["sex"] == "Female", "total_bill"].mean() # 18.056896551724137
total_bill_male = df.loc[df["sex"] == "Male", "total_bill"].mean() # 20.744076433121034


def total_bills(sex, total_bill):
    if sex == "Female":
        if total_bill > total_bill_fem:
            return 1
        else:
            return 0
    if sex == "Male":
        if total_bill > total_bill_male:
            return 1
        else:
            return 0

total_bills("Female", 5)

df["total_bill_flag"] = df.apply(lambda x: total_bills(x["sex"], x["total_bill"]), axis=1)
df.head()

# Görev 24: total_bill_flag değişkenini kullanarak
# cinsiyetlere göre ortalamanın altında ve üstünde olanların sayısını gözlemleyiniz.

df["total_bill_flag"].value_counts()
df.groupby("sex")["total_bill_flag"].value_counts()
df.groupby(["sex", "total_bill_flag"]).agg({"total_bill_flag": "count"}) #hoca


# Görev 25: Veriyi total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve
# ilk 30 kişiyi yeni bir dataframe'e atayınız.

df_new = df.sort_values(by='total_bill_tip_sum', ascending=False).head(30)
df_new
