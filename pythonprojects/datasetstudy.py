import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
df=pd.read_csv(
    r"--your file path--",
    encoding="utf-8-sig"
)

#Lets clean the dataset! We have some prices in Lakh, some in Cr, so lets make them the same for better analysis!
def convert_price(x):
    x=str(x).strip().lower()
    try:
        if "lac" in x:
            return float(x.split()[0]) * 1e5   
        elif "cr" in x or "crore" in x:
            return float(x.split()[0]) * 1e7   
        else:
            return None
    except:
        return None

df["price_clean"]=df["price"].apply(convert_price)

#Lets clean the dataset! This time, we clean the area parameter.

df["area_clean"]=df["area"].str.extract(r"(\d+\.?\d*)").astype(float)

print(df[["price", "price_clean", "area", "area_clean"]].head())

#Clean again! The bedRoom column has values: ['2 Bedrooms' '3 Bedrooms' '4 Bedrooms' '1 Bedroom' '5 Bedrooms' 'bedRoom'
# '6 Bedrooms' nan], lets remove the 'nan' for better results!

df = df[df["bedRoom"].str.contains(r"\d", na=False)]
df["bedRoom_clean"] = df["bedRoom"].str.extract(r"(\d+)").astype(int)
bed_price = df.groupby("bedRoom_clean")["price_clean"].mean()

# Plot 1: Price Distribution

plt.hist(df["price_clean"].dropna()/1e5, bins=20, color="pink")
plt.xlabel("Price (in Lakhs)")
plt.ylabel("Number of Flats")
plt.title("🌸 Distribution of Flat Prices 🌸")
plt.show()


# Plot 2: Bedrooms vs Avg Price

bed_price = df.groupby("bedRoom")["price_clean"].mean() / 1e5  

bed_price.plot(kind="bar", color=["#ffb6c1", "#ff69b4", "#db7093"])
plt.xlabel("Number of Bedrooms")
plt.ylabel("Average Price (Lakhs)")
plt.title("🌸 Average Price by Bedrooms 🌸")
plt.show()

# Plot 3: Area vs Price (Scatter)

plt.scatter(df["area_clean"], df["price_clean"]/1e5, alpha=0.5, color="purple")
plt.xlabel("Area (sq.ft.)")
plt.ylabel("Price (Lakhs)")
plt.title("🌸 Area vs Price 🌸")
plt.show()


#If you've reached till the end, congratulations! Here's a song recommendation: Supernatural by NewJeans🐰