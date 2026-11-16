#!/usr/bin/env python
# coding: utf-8

# ## Analysis of Retail Sales Data

# ### OVERVIEW

# ###### • Dataset Description:
# ###### • Order_ID: Unique identifier of the order.
# ###### • Order_Date: Date on which the order was placed.
# ###### • Customer_ID: Unique identifier of the customer.
# ###### • Customer_Age: Age of the customer.
# ###### • Gender: Gender of the customer.
# ###### • Region: Region where the order was placed.
# ###### • City: City of the customer/order.
# ###### • Category: Main product category.
# ###### • Sub_Category: Product sub-category.
# ###### • Product: Product purchased.
# ###### • Quantity: Number of products purchased.
# ###### • Unit_Price: Price of one unit.
# ###### • Sales: Total sales amount.
# ###### • Discount: Discount given on the order.
# ###### • Payment_Mode: Payment method used.
# ###### • Customer_Type: Type of customer.
# ###### • Order_Status: Current status of the order.
# ###### • Rating: Customer rating.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("business_retail_data.csv")
df


# #### UNDERSTANDING THE DATA

# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


df.dtypes


# In[9]:


df.info()


# In[10]:


df.describe()


# #### CHECKING FOR ERRORS

# In[11]:


df.isnull().sum()


# In[12]:


df["Customer_Age"] = df["Customer_Age"].fillna(df["Customer_Age"].median())
df["City"] = df["City"].fillna("Unknown")
df["Discount"] = df["Discount"].fillna(df["Discount"].median())
df["Payment_Mode"] = df["Payment_Mode"].fillna("Unknown")
df["Rating"] = df["Rating"].fillna(df["Rating"].median())


# In[13]:


df.duplicated().sum()


# In[14]:


df = df.drop_duplicates()


# In[15]:


df["Order_Date"] = pd.to_datetime(df["Order_Date"])


# In[16]:


df.isnull().sum()
df.duplicated().sum()
df.info()


# #### UNIVARIATE ANALYSIS

# In[17]:


sns.histplot(data=df,x="Customer_Age")
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.show()


# ##### Customers belong to different age groups, showing that the business attracts people of different ages.

# In[18]:


sns.countplot(data=df, x="Gender")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.show()


# ##### Female customers have the highest representation,followed by male customers,other genders make up smaller portion of customers. This indicates a slightly female-dominated customer base.

# In[19]:


sns.countplot(data=df,x="Category")
plt.title("Orders by Category")
plt.xlabel("Category")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.show()


# ##### Electronics has the highest number of orders, followed by Clothing and Home. Sports has the lowest number of orders.

# In[20]:


sns.countplot(data=df,x="Region")
plt.title("Orders by Region")
plt.xlabel("Region")
plt.ylabel("Number of Orders")
plt.show()


# ##### North has the highest number of orders, while East has the lowest. Overall, the orders are fairly balanced across the regions.

# In[21]:


sns.countplot(data=df,x="Payment_Mode")
plt.title("Payment Mode Distribution")
plt.xticks(rotation=45)
plt.show()


# ##### UPI clearly leads,followed by Credit Card and Debit Card. Cash is the least-used payment mode, highlighting a strong preference for digital payments.

# In[22]:


sns.countplot(data=df, x="Order_Status")
plt.title("Order Status Distribution")
plt.xlabel("Order Status")
plt.ylabel("Number of Orders")
plt.show()


# ##### Most orders are delivered successfully. Cancelled, Pending, and Returned orders make up a smaller portion of the total orders.

# In[23]:


sns.countplot(data=df,x="Rating")
plt.title("Customer Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")
plt.show()


# ##### 5-star ratings are the most common, showing that many customers are satisfied with their purchases.

# In[24]:


sns.histplot(data=df,x="Shipping_Days",discrete=True)
plt.title("Shipping Days Distribution")
plt.xlabel("Shipping Days")
plt.ylabel("Number of Orders")
plt.show()


# ##### Most orders take between 1 and 10 days to deliver, with 6 days being the most common delivery time.

# In[25]:


sns.countplot(data=df,x="City")
plt.title("Customer Distribution by City")
plt.xlabel("City")
plt.ylabel("Number of Customers")
plt.xticks(rotation=60)
plt.show()


# ##### Kolkata has the highest number of customers, followed by Chandigarh and Chennai. This shows that customers are spread across different cities.

# In[26]:


sns.histplot(data=df,x="Profit",bins=30,kde=True)
plt.title("Profit Distribution")
plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.show()


# ##### Profit values are different for each order, with most orders having low to medium profit.

# In[27]:


sns.histplot(data=df,x="Sales",kde=True)
plt.title("Sales Distribution")
plt.show()


# ##### Most orders have lower sales values, while only a few orders have very high sales amounts.

# In[28]:


payment_counts = df["Payment_Mode"].value_counts()
plt.pie(payment_counts, labels=payment_counts.index, autopct="%1.1f%%")
plt.title("Payment Mode Distribution")
plt.show()


# ##### UPI has the largest share of orders at around 37.2%, while Cash has the smallest share.

# In[29]:


customer_type_counts = df["Customer_Type"].value_counts()
plt.pie(customer_type_counts, labels=customer_type_counts.index,autopct="%1.1f%%")
plt.title("Customer Type Distribution")
plt.show()


# ##### Regular customers form the largest group, followed by Premium and New customers. Corporate customers form the smallest group.

# In[30]:


category_counts = df["Category"].value_counts()
plt.pie(category_counts,labels=category_counts.index,autopct="%1.1f%%")
plt.title("Category Distribution")
plt.show()


# ##### Electronics has the largest share of orders at around 28.4%, while Sports has the smallest share.

# In[31]:


region_counts = df["Region"].value_counts()
plt.pie(region_counts,labels=region_counts.index,autopct="%1.1f%%")
plt.title("Region Distribution")
plt.show()


# ##### The order share is almost equal across all regions. North has the highest share, while East has the lowest.

# ## BIVARIATE ANALYSIS

# In[32]:


sns.barplot(data=df,x="Category",y="Sales",estimator="sum")
plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.show()


# ##### Electronics has the highest sales, with around ₹95.7M. This shows that Electronics is the best-selling category.

# In[33]:


sns.barplot(data=df,x="Category",y="Profit")
plt.title("Total Profit by Category")
plt.xlabel("Category")
plt.ylabel("Profit")
plt.xticks(rotation=45)
plt.show()


# ##### Electronics gives the highest profit at around ₹25.5M. Beauty has the lowest profit among the categories.

# In[34]:


sns.barplot(data=df,x="Region",y="Sales")
plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.show()


# ##### West has the highest sales at around ₹32.4M, while South has the lowest sales.

# In[35]:


sns.barplot(data=df, x="Customer_Type", y="Sales",estimator="sum")
plt.title("Total Sales by Customer Type")
plt.xlabel("Customer Type")
plt.ylabel("Total Sales")
plt.show()


# ##### Regular customers generate the highest sales at around ₹58.4M. This shows that Regular customers are an important customer group for the business.

# In[36]:


sns.barplot(data=df,x="Customer_Type",y="Profit",estimator="sum")
plt.title("Total Profit by Customer Type")
plt.xlabel("Customer Type")
plt.ylabel("Total Profit")
plt.show()


# ##### Regular customers contribute the highest profit, at around ₹15.8M. Corporate customers contribute the least profit.

# In[37]:


sns.barplot(data=df,x="Gender",y="Sales",estimator="sum")
plt.title("Total Sales by Gender")
plt.xlabel("Gender")
plt.ylabel("Total Sales")
plt.show()


# ##### Female customers generate slightly higher sales than male customers. This shows that female customers contribute a little more to total sales.

# In[38]:


sns.barplot(data=df,x="Gender",y="Quantity", estimator="sum")
plt.title("Total Quantity Purchased by Gender")
plt.xlabel("Gender")
plt.ylabel("Quantity")
plt.show()


# ##### Female customers purchase the highest quantity, followed by Male customers. Other genders have the lowest quantity.

# In[39]:


sns.barplot(data=df, x="Payment_Mode", y="Sales",estimator="sum")
plt.title("Total Sales by Payment Mode")
plt.xlabel("Payment Mode")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()


# ##### UPI generates the highest sales at around ₹48.7M. Cash has the lowest sales among the payment methods.

# In[40]:


sns.barplot(data=df,x="Category",y="Quantity")
plt.title("Category vs Quantity")
plt.xlabel("Category")
plt.ylabel("Total Quantity")
plt.xticks(rotation=45)
plt.show()


# ##### Electronics has the highest quantity purchased, while Sports has the lowest. This shows that Electronics products have higher demand.

# In[41]:


sns.barplot(data=df, x="Region", y="Quantity", estimator="sum")
plt.title("Total Quantity by Region")
plt.xlabel("Region")
plt.ylabel("Total Quantity")
plt.show()


# ##### North has the highest quantity purchased, while East has the lowest. The difference between the regions is not very large.

# In[42]:


sns.barplot(x="Product",y="Quantity",data=df)
plt.title("Product vs Quantity")
plt.xlabel("Product")
plt.ylabel("Quantity")
plt.xticks(rotation=90)
plt.show()


# ##### The quantity purchased is different for each product. Some products have much higher demand than others.

# In[43]:


sns.scatterplot(data=df,x="Sales",y="Profit")
plt.title("Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.show()


# ##### Sales and profit generally increase together. This means that orders with higher sales usually generate higher profit.

# In[44]:


sns.scatterplot(data=df,x="Discount",y="Profit")
plt.title("Discount vs Profit")
plt.xlabel("Discount")
plt.ylabel("Profit")
plt.show()


# ##### Profit changes at different discount levels. This shows that giving higher discounts can affect the profit earned from an order.

# In[45]:


sns.scatterplot(data=df, x="Quantity", y="Sales")
plt.title("Quantity vs Sales")
plt.xlabel("Quantity")
plt.ylabel("Sales")
plt.show()


# ##### Orders with higher quantities generally have higher sales. This shows that quantity is an important factor affecting sales.

# In[46]:


sns.scatterplot(data=df, x="Customer_Age", y="Sales")
plt.title("Customer Age vs Sales")
plt.xlabel("Age")
plt.ylabel("Sales")
plt.show()


# ##### Sales are spread across different age groups. There is no single age group that clearly dominates the sales.

# In[47]:


sns.scatterplot(data=df, x="Unit_Price", y="Quantity")
plt.title("Unit Price vs Quantity")
plt.xlabel("Unit Price")
plt.ylabel("Quantity")
plt.show()


# ##### Customers purchase different quantities at different prices.

# In[48]:


sns.scatterplot(data=df,x="Shipping_Days",y="Rating")
plt.title("Shipping Days vs Customer Rating")
plt.xlabel("Shipping Days")
plt.ylabel("Rating")
plt.show()


# ##### Customer ratings are different across shipping times. This helps us understand whether delivery time affects customer satisfaction.

# In[49]:


sns.boxplot(data=df,x="Category",y="Rating")
plt.title("Rating by Category")
plt.xticks(rotation=45)
plt.show()


# ##### Customer ratings are different across categories. This helps compare customer satisfaction for different products.

# In[50]:


sns.boxplot(x="Order_Status",y="Shipping_Days",data=df)
plt.title("Shipping Days by Order Status")
plt.xlabel("Order Status")
plt.ylabel("Shipping Days")
plt.show()


# ##### Shipping time varies between different order statuses. Delivered, cancelled, pending, and returned orders show different shipping patterns.

# In[51]:


sns.boxplot(data=df,x="Customer_Type",y="Rating")
plt.title("Rating by Customer Type")
plt.xlabel("Customer Type")
plt.ylabel("Rating")
plt.show()


# ##### Customer ratings vary between different customer types. This helps us understand which customer group is more satisfied.

# In[52]:


sns.boxplot(data=df, x="Category", y="Shipping_Days")
plt.title("Shipping Days by Category")
plt.xlabel("Category")
plt.ylabel("Shipping Days")
plt.xticks(rotation=45)
plt.show()


# ##### Different categories have different delivery times. Some products take longer to reach customers than others.

# In[53]:


sns.boxplot(data=df, x="Region", y="Shipping_Days")
plt.title("Shipping Days by Region")
plt.xlabel("Region")
plt.ylabel("Shipping Days")
plt.show()


# ##### Delivery time varies across regions.This may be because of differences in distance and delivery conditions.

# In[54]:


year = df["Order_Date"].dt.year
sales = df.groupby(year)["Sales"].sum()
plt.plot(sales.index, sales.values, marker="o")
plt.title("Yearly Sales")
plt.xlabel("Year")
plt.ylabel("Total Sales")
plt.xticks(sales.index)
plt.show()


# ##### Sales decreased from around ₹44.2M in 2023 to ₹38.95M in 2025. This shows a decrease in sales over the three years.

# In[55]:


year = df["Order_Date"].dt.year
profit = df.groupby(year)["Profit"].sum()
plt.plot(profit.index, profit.values, marker="o")
plt.title("Yearly Profit")
plt.xlabel("Year")
plt.ylabel("Total Profit")
plt.xticks(profit.index)
plt.show()


# ##### Profit decreased from approximately ₹11.6M in 2023 to ₹10.7M in 2025, showing a gradual decline in overall profitability.

# In[56]:


df["Order_Date"] = pd.to_datetime(df["Order_Date"])
month = df["Order_Date"].dt.to_period("M")
sales = df.groupby(month)["Sales"].sum()
plt.figure(figsize=(12,5))
plt.plot(sales.index.astype(str), sales.values, marker="o")
plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()


# ##### Monthly sales go up and down throughout the year.This shows that sales change from month to month and may have some seasonal patterns.

# In[57]:


city_sales = df.groupby("City")["Sales"].sum()
top_10 = city_sales.sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
plt.bar(top_10.index, top_10.values)

plt.title("Top 10 Cities by Sales")
plt.xlabel("City")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)
plt.show()


# ##### Pune leads the top 10 cities in sales,followed by Chandigarh and Kochi. This identifies Pune as the strongest city-level market.

# #### Conclusion
# ##### • EDA helped identify important sales and customer trends.
# ##### • Electronics is the strongest-performing category.
# ##### • Regular customers contribute the most sales and profit.
# ##### • UPI is the most preferred payment method.
# ##### • Sales and profit show a declining trend over the years.
# ##### • These insights can help the business make better decisions and improve performance.

# #### Suggestions
# ##### • Focus more on Electronics, as it has the highest sales and profit.
# ##### • Increase promotions for low-performing categories like Sports.
# ##### • Encourage UPI and other digital payments.
# ##### • Give loyalty offers to Regular customers.
# ##### • Control discounts to maintain better profit margins.
# ##### • Improve delivery speed in regions with longer shipping times.
# ##### • Focus on high-performing cities like Pune.
# ##### • Develop new strategies to improve the declining yearly sales and profit.
