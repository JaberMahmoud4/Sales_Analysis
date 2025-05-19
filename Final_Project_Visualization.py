# Import necessary libraries for data manipulation, visualization, and clustering
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For plotting charts
import seaborn as sns  # For enhanced data visualization
from sklearn.cluster import KMeans  # For performing K-Means clustering
import numpy as np  # For numerical operations
import plotly.graph_objects as go  # For interactive visualizations
import squarify  # For creating treemaps

# Load the dataset to understand its structure and columns
data = pd.read_csv('sales_data_sample.csv', encoding='ISO-8859-1')  # Read the sales dataset using pandas

# Convert 'ORDERDATE' column to datetime format to enable time-based operations
data['ORDERDATE'] = pd.to_datetime(data['ORDERDATE'], errors='coerce')  
# Remove rows where 'ORDERDATE' could not be converted to datetime (e.g., missing or invalid dates)
data.dropna(subset=['ORDERDATE'], inplace=True)
# Set 'ORDERDATE' as the index of the DataFrame for easy time-based analysis
data.set_index('ORDERDATE', inplace=True)

# K-Means Clustering on Quantity Ordered and Sales
kmeans_data = data[['QUANTITYORDERED', 'SALES']].dropna()  # Extract relevant columns and remove rows with missing values
kmeans = KMeans(n_clusters=3, random_state=0).fit(kmeans_data)  # Perform K-Means clustering with 3 clusters
data['Cluster'] = kmeans.labels_  # Assign the cluster labels to a new column in the DataFrame

# Visualization 1: Monthly Sales Trend
plt.figure(figsize=(10, 6))  # Set the size of the plot
monthly_sales = data['SALES'].resample('ME').sum()  # Resample sales data to calculate monthly totals ('ME' stands for month-end)
plt.plot(monthly_sales.index, monthly_sales.values, color='blue', label='Sales')  # Plot the monthly sales trend
plt.fill_between(monthly_sales.index, monthly_sales.values, color='blue', alpha=0.3)  # Add a shaded area under the line for emphasis
plt.title('Monthly Sales Trend')  # Title of the plot
plt.xlabel('Date')  # Label for the x-axis
plt.ylabel('Sales ($)')  # Label for the y-axis
plt.legend()  # Add a legend to the plot

# Add an annotation to highlight observations about the sales trend
plt.text(
    pd.to_datetime("2004-01"),  # Position of the annotation on the x-axis
    monthly_sales.max() * 0.8,  # Position of the annotation on the y-axis
    "Observe periodic fluctuations,\nindicating seasonal patterns.\nNotable peak in Q4 likely due\nto holiday sales boost.", 
    color="darkblue", fontsize=10, bbox=dict(facecolor="white", alpha=0.5)
)  # Custom text with styling

# Display the plot
plt.show()


# Scatter plot

# Visualization 2: Linear Regression of Price Each vs Sales with DEALSIZE categories
plt.figure(figsize=(10, 6))  # Set the size of the figure
sns.scatterplot(
    x='PRICEEACH', y='SALES', data=data, hue='DEALSIZE', 
    palette='coolwarm', alpha=0.6
)  # Scatter plot of Price Each vs Sales with points colored by DEALSIZE
sns.regplot(
    x='PRICEEACH', y='SALES', data=data, scatter=False, 
    color='blue', line_kws={'alpha': 0.5}
)  # Add a linear regression line without scatter points
plt.title('Linear Regression of Price Each vs Sales')  # Title of the plot
plt.xlabel('Price Each ($)')  # Label for the x-axis
plt.ylabel('Sales ($)')  # Label for the y-axis
plt.legend(title='Deal Size')  # Add a legend for DEALSIZE categories
plt.show()  # Display the plot

# Visualization 3: Quantity Ordered vs Sales with K-Means Clustering and Linear Regression
plt.figure(figsize=(10, 6))  # Set the size of the figure
sns.scatterplot(
    x='QUANTITYORDERED', y='SALES', data=data, hue='Cluster', 
    palette='viridis', alpha=0.6
)  # Scatter plot of Quantity Ordered vs Sales with points colored by clusters
sns.regplot(
    x='QUANTITYORDERED', y='SALES', data=data, scatter=False, 
    color='darkgreen', line_kws={'alpha': 0.5}
)  # Add a linear regression line
plt.title('Quantity Ordered vs Sales with K-Means Clustering and Linear Regression')
plt.xlabel('Quantity Ordered')  # Label for the x-axis
plt.ylabel('Sales ($)')  # Label for the y-axis
plt.legend(title='Cluster')  # Add a legend for clusters
plt.show()  # Display the plot

# Visualization 4: Sales vs Price Each by Product Line (FacetGrid)
g = sns.FacetGrid(
    data, col="PRODUCTLINE", hue="DEALSIZE", palette="viridis", 
    col_wrap=3, height=4
)  # Create a grid of scatter plots for each PRODUCTLINE, colored by DEALSIZE
g.map(plt.scatter, "PRICEEACH", "SALES", alpha=0.7, s=50)  # Plot scatter points
g.add_legend(title='Deal Size')  # Add a legend for DEALSIZE
g.set_axis_labels("Price Each ($)", "Sales ($)")  # Set axis labels for all plots
g.fig.suptitle('Sales vs Price Each by Product Line')  # Add a main title for the grid
g.fig.subplots_adjust(top=0.9)  # Adjust spacing to fit the title
plt.show()  # Display the grid of plots

# Visualization 5: 3D Scatter Plot of Quantity Ordered vs Sales vs Price Each
fig = plt.figure(figsize=(10, 6))  # Set the figure size
ax = fig.add_subplot(111, projection='3d')  # Create a 3D plot
ax.scatter(
    data['QUANTITYORDERED'], data['SALES'], data['PRICEEACH'], 
    c=data['Cluster'], cmap='viridis', alpha=0.6
)  # Plot 3D points with colors based on clusters
ax.set_xlabel('Quantity Ordered')  # Label for the x-axis
ax.set_ylabel('Sales ($)')  # Label for the y-axis
ax.set_zlabel('Price Each ($)')  # Label for the z-axis
ax.set_title('3D Scatter Plot of Quantity Ordered vs Sales vs Price Each')  # Add a title
plt.show()  # Display the 3D plot

# Bubble chart

# Visualization 6: Sales Distribution by Order Quantity (Bubble Chart with K-Means Clusters)
plt.figure(figsize=(10, 6))  # Set the figure size
sizes = data['QUANTITYORDERED']  # Determine bubble sizes based on Quantity Ordered
sns.scatterplot(
    x='QUANTITYORDERED', y='SALES', data=data, hue='Cluster', 
    size=sizes, sizes=(20, 200), alpha=0.6, palette='viridis'
)  # Create a bubble chart where bubble sizes represent Quantity Ordered
plt.title('Bubble Chart with K-Means Clusters (Quantity Ordered vs Sales)')  # Title of the plot
plt.xlabel('Quantity Ordered')  # Label for the x-axis
plt.ylabel('Sales ($)')  # Label for the y-axis
plt.legend(title='Cluster')  # Add a legend for clusters
plt.show()  # Display the bubble chart


# Bar chart

# Visualization 7: Sales by Product Category with Deal Size Segmentation
plt.figure(figsize=(10, 6))  # Set the figure size
product_sales = data.groupby(['PRODUCTLINE', 'DEALSIZE'])['SALES'].sum().unstack()  # Group data by PRODUCTLINE and DEALSIZE, summing SALES
product_sales.plot(kind='bar', stacked=True, colormap='viridis')  # Create a stacked bar chart
plt.title('Total Sales by Product Line and Deal Size')  # Title of the chart
plt.xlabel('Product Line')  # Label for the x-axis
plt.ylabel('Sales ($)')  # Label for the y-axis
plt.legend(title='Deal Size', bbox_to_anchor=(1.05, 1))  # Add a legend for Deal Size, placing it outside the plot
plt.text(-0.5, product_sales.max().max() * 0.7, 
         "Product lines like Classic Cars and Motorcycles\nshow larger deal sizes. Note small deals in\nproduct lines such as Planes and Ships.", 
         color="black", fontsize=10, bbox=dict(facecolor="white", alpha=0.5))  # Add explanatory text
plt.show()  # Display the bar chart

# Visualization 8: Sales by Country and Territory
plt.figure(figsize=(12, 8))  # Set the figure size
country_sales = data.groupby(['COUNTRY', 'TERRITORY'])['SALES'].sum().unstack()  # Group data by COUNTRY and TERRITORY, summing SALES
country_sales.plot(kind='bar', stacked=True, colormap='Spectral')  # Create a stacked bar chart
plt.title('Sales by Country and Territory')  # Title of the chart
plt.xlabel('Country')  # Label for the x-axis
plt.ylabel('Sales ($)')  # Label for the y-axis
plt.legend(title='Territory', bbox_to_anchor=(1.05, 1))  # Add a legend for Territory, placing it outside the plot
plt.text(-0.5, country_sales.max().max() * 0.8, 
         "Highest sales from US and European regions.\nEMEA territories show diverse contributions,\nindicating strong regional presence.", 
         color="black", fontsize=10, bbox=dict(facecolor="white", alpha=0.5))  # Add explanatory text
plt.show()  # Display the bar chart

# Pie chart

# Visualization 9: Distribution of Deal Sizes
plt.figure(figsize=(8, 8))  # Set the figure size
deal_size_counts = data['DEALSIZE'].value_counts()  # Count occurrences of each DEALSIZE
plt.pie(
    deal_size_counts, labels=deal_size_counts.index, autopct='%1.1f%%', 
    startangle=140, colors=sns.color_palette("pastel")
)  # Create a pie chart with percentage labels
plt.title('Deal Size Distribution')  # Title of the chart
plt.text(-1.5, 1.5, 
         "Largest share is small deals, suggesting\nmost orders are low-volume.\nMedium deals also hold a strong presence.", 
         color="black", fontsize=10, bbox=dict(facecolor="white", alpha=0.5))  # Add explanatory text
plt.show()  # Display the pie chart

# Correlation Heatmap

# Visualization 10: Correlation Heatmap
plt.figure(figsize=(10, 8))  # Set the figure size
numeric_cols = ['QUANTITYORDERED', 'PRICEEACH', 'ORDERLINENUMBER', 'SALES', 'QTR_ID', 'MONTH_ID', 'YEAR_ID', 'MSRP']  # Select numeric columns
correlation_matrix = data[numeric_cols].corr()  # Compute the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')  # Create a heatmap with annotations
plt.title('Correlation Heatmap')  # Title of the heatmap
plt.text(1.5, -0.7, 
         "High correlation between\nQuantity Ordered and Sales.\nWeak correlation between MSRP and Sales,\nsuggesting MSRP doesn’t strongly affect revenue.", 
         color="black", fontsize=10, bbox=dict(facecolor="white", alpha=0.5))  # Add explanatory text
plt.show()  # Display the heatmap

# Histogram

# Visualization 11: Sales Distribution Histogram
plt.figure(figsize=(10, 6))  # Set the figure size
sns.histplot(
    data['SALES'], bins=50, kde=True, color='purple', alpha=0.7
)  # Create a histogram with KDE (Kernel Density Estimation)
plt.title('Sales Distribution Histogram')  # Title of the histogram
plt.xlabel('Sales ($)')  # Label for the x-axis
plt.ylabel('Frequency')  # Label for the y-axis
plt.axvline(
    data['SALES'].mean(), color='red', linestyle='dashed', linewidth=1, 
    label=f"Mean: ${data['SALES'].mean():.2f}"
)  # Add a vertical line for the mean
plt.axvline(
    data['SALES'].median(), color='blue', linestyle='dashed', linewidth=1, 
    label=f"Median: ${data['SALES'].median():.2f}"
)  # Add a vertical line for the median
plt.legend()  # Add a legend for the mean and median lines
plt.text(data['SALES'].max() * 0.7, 100, 
         "The distribution shows most sales are in\nthe lower range, with a few high-value outliers.\nKDE curve helps identify peaks clearly.", 
         color="black", fontsize=10, bbox=dict(facecolor="white", alpha=0.5))  # Add explanatory text
plt.show()  # Display the histogram

# Radar Chart

# Visualization 12: Radar Chart for Product Lines

# Extract unique product lines from the data for the radar chart labels
labels = data['PRODUCTLINE'].unique()

# Calculate total sales for each product line
stats = [data[data['PRODUCTLINE'] == label]['SALES'].sum() for label in labels]

# Generate angles for the radar chart, ensuring a circular layout
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

# Close the loop for the radar chart by appending the first element to the end
stats += stats[:1]  # Repeat the first value to close the chart
angles += angles[:1]  # Repeat the first angle to close the chart

# Create a polar plot for the radar chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Fill the radar chart area with a transparent red color
ax.fill(angles, stats, color='red', alpha=0.25)

# Remove the y-axis labels for a cleaner look
ax.set_yticklabels([])

# Set the positions of the labels around the radar chart
ax.set_xticks(angles[:-1])

# Apply the product line labels to the positions
ax.set_xticklabels(labels)

# Add a title to the radar chart
plt.title('Radar Chart of Total Sales by Product Line')

# Display the radar chart
plt.show()


# Area Chart

# Visualization 13: Area Chart of quarterly sales trends with cumulative growth visible over time.

# Create a figure for the area chart with a defined size
plt.figure(figsize=(10, 6))

# Resample sales data to calculate total sales at the end of each quarter (QE = Quarter End)
quarterly_sales = data['SALES'].resample('QE').sum()

# Create a filled area plot for quarterly sales with a semi-transparent sky blue color
plt.fill_between(quarterly_sales.index, quarterly_sales.values, color='skyblue', alpha=0.5)

# Plot the sales line on top of the area plot with a solid blue line
plt.plot(quarterly_sales.index, quarterly_sales.values, color='blue', linewidth=2)

# Add a title to the area chart
plt.title('Quarterly Sales Area Chart')

# Label the x-axis as 'Date'
plt.xlabel('Date')

# Label the y-axis as 'Sales ($)'
plt.ylabel('Sales ($)')

# Add a text annotation highlighting trends in the data
plt.text(quarterly_sales.index[1], quarterly_sales.max() * 0.8,
         "Clear growth trend over quarters.\nQ4 shows the most significant peak,\nindicating holiday boosts.",
         fontsize=10, bbox=dict(facecolor="white", alpha=0.5))

# Display the area chart
plt.show()

# Box Plot

# Visualization 14: Box Plot of sales by product line.
# Filter the data to exclude the top 25% of sales outliers for a clearer analysis of variability
filtered_sales = data[(data['SALES'] < data['SALES'].quantile(0.75))]  # Removing top 25% outliers

# Set the figure size for better visibility
plt.figure(figsize=(10, 6))

# Create a box plot showing the distribution of sales across product lines
sns.boxplot(x='PRODUCTLINE', y='SALES', data=filtered_sales, hue='PRODUCTLINE', legend=False)

# Add title and axis labels for the plot
plt.title('Box Plot of Sales by Product Line (Outliers Removed)')
plt.xlabel('Product Line')
plt.ylabel('Sales ($)')

# Add a descriptive annotation to explain the plot's insights
plt.text(-0.5, filtered_sales['SALES'].max() * 0.7,
         "Outliers removed (top 25%).\nMotorcycles and Classic Cars show\nhigher sales variability.",
         fontsize=10, bbox=dict(facecolor="white", alpha=0.5))

# Display the box plot
plt.show()

# Violin Chart

# Visualization 15: Violin Chart highlighting the distribution and density of sales data for each product line.

# Set the figure size for better visibility
plt.figure(figsize=(10, 6))

# Create a violin plot showing sales data distribution for product lines
sns.violinplot(x='PRODUCTLINE', y='SALES', data=filtered_sales, inner='quartile', hue='PRODUCTLINE', legend=False)

# Add title and axis labels for the plot
plt.title('Violin Plot of Sales by Product Line (Outliers Removed)')
plt.xlabel('Product Line')
plt.ylabel('Sales ($)')

# Add a descriptive annotation to explain the plot's insights
plt.text(-0.5, filtered_sales['SALES'].max() * 0.7,
         "Violin plot reveals distribution\nand density of sales values\nwithin each product line.",
         fontsize=10, bbox=dict(facecolor="white", alpha=0.5))

# Display the violin plot
plt.show()

# Spider Chart

# Visualization 16: Spider Chart to compare sales across product lines for different clusters.

# Group the data by 'Cluster' and 'Product Line', then sum up the sales for each group
cluster_sales = data.groupby(['Cluster', 'PRODUCTLINE'])['SALES'].sum().unstack().fillna(0)

# Extract product line names as labels for the chart
labels = cluster_sales.columns

# Define the angles for the spider chart (one for each product line)
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # Close the loop to make the spider chart circular

# Set the figure size for better visibility
plt.figure(figsize=(10, 10))

# Iterate through each cluster and plot its sales distribution on the spider chart
for cluster, row in cluster_sales.iterrows():
    # Convert sales data to a list and close the loop
    stats = row.tolist()
    stats += stats[:1]  # Close the loop for circular plotting
    
    # Plot the sales data for the current cluster
    plt.polar(angles, stats, label=f'Cluster {cluster}')

# Add an average line to show the mean sales across all clusters
plt.fill(angles, cluster_sales.mean(axis=0).tolist() + [cluster_sales.mean(axis=0).tolist()[0]],
         color='gray', alpha=0.2, label='Average')

# Add title and labels for the spider chart
plt.title('Spider Chart: Sales Distribution by Product Line and Cluster', fontsize=14)
plt.xticks(angles[:-1], labels)  # Set the labels for the axes
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))  # Position the legend

# Display the spider chart
plt.show()


# Doughnut Chart

# Visualization 17: Deal Size Distribution as Doughnut Chart
plt.figure(figsize=(8, 8))  # Set figure size for the doughnut chart
deal_size_counts = data['DEALSIZE'].value_counts()  # Count the occurrences of each deal size
# Plot a pie chart with specific parameters to create the doughnut chart
plt.pie(deal_size_counts, 
        labels=deal_size_counts.index, 
        autopct='%1.1f%%',  # Format percentage with 1 decimal place
        startangle=140,  # Rotate the chart for better appearance
        colors=sns.color_palette("pastel"),  # Use pastel color palette
        wedgeprops=dict(width=0.3))  # Make the pie chart into a doughnut by setting width
plt.title('Deal Size Distribution (Doughnut Chart)')  # Add a title
# Add annotation text to the chart to provide additional insights
plt.text(-1.5, 1.5, 
         "Largest share is small deals, suggesting\nmost orders are low-volume.\nMedium deals also hold a strong presence.", 
         color="black", fontsize=10, bbox=dict(facecolor="white", alpha=0.5))
plt.show()  # Display the chart

# Gauge Chart 

# Visualization 18: Sales Performance Gauge
# We’ll use Plotly to create a gauge chart for Sales Performance
sales_performance = data['SALES'].sum()  # Calculate the total sales
max_sales = 100000000  # Set an arbitrary maximum value for the gauge
# Create the gauge chart using Plotly
gauge_fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",  # Display mode includes gauge, number, and delta
    value=sales_performance,  # Current sales performance value
    delta={'reference': 50000000},  # Reference value for delta comparison
    gauge={'axis': {'range': [0, max_sales]},  # Set range for the gauge axis
           'bar': {'color': "darkblue"},  # Color of the gauge bar
           'steps': [  # Define color steps for different ranges
               {'range': [0, max_sales * 0.4], 'color': "lightgray"},
               {'range': [max_sales * 0.4, max_sales * 0.7], 'color': "lightyellow"},
               {'range': [max_sales * 0.7, max_sales], 'color': "lightgreen"}]},
    title={'text': "Total Sales Performance ($)", 'font': {'size': 20}}  # Set title and font size
))
gauge_fig.show()  # Display the gauge chart

# Comparison Chart 

# Visualization 19: Sales Comparison Between Countries (Side-by-side Bar Chart)
plt.figure(figsize=(10, 6))  # Set figure size for the bar chart
country_sales = data.groupby(['COUNTRY'])['SALES'].sum()  # Calculate total sales per country
top_countries = country_sales.nlargest(10)  # Get top 10 countries by sales
other_countries = country_sales.drop(top_countries.index)  # Get remaining countries

# Create the comparison chart: Side-by-side bars for the top 10 countries and others combined
plt.bar(top_countries.index, top_countries.values, color='skyblue', label='Top 10 Countries')  # Plot top 10 countries
plt.bar(['Others'], [other_countries.sum()], color='lightcoral', label='Other Countries')  # Plot other countries combined
plt.title('Sales Comparison: Top 10 Countries vs Other Countries')  # Add a title
plt.xlabel('Country')  # X-axis label
plt.ylabel('Sales ($)')  # Y-axis label
plt.xticks(rotation=45, ha='right')  # Rotate X-axis labels for readability
plt.legend()  # Add legend for clarity
# Add annotation text to provide insights on the chart
plt.text(2, top_countries.max() * 0.8, 
         "The largest sales come from a few countries,\nwhile the others contribute to a smaller share.", 
         color="black", fontsize=10, bbox=dict(facecolor="white", alpha=0.5))
plt.show()  # Display the bar chart

# TreeMap Chart

# Visualization 20: TreeMap of Sales by Product Line
# Calculate total sales by product line
sales_by_productline = data.groupby('PRODUCTLINE')['SALES'].sum().reset_index()

# Sort the product lines by sales in descending order
sales_by_productline = sales_by_productline.sort_values('SALES', ascending=False)

# Create a tree map to visualize the sales distribution across product lines
plt.figure(figsize=(10, 8))  # Set figure size for the tree map
squarify.plot(sizes=sales_by_productline['SALES'],  # Plot the sizes based on sales
              label=sales_by_productline['PRODUCTLINE'] + '\n' + sales_by_productline['SALES'].apply(lambda x: f'${x:,.0f}'),  # Label with product line and sales amount
              color=sns.color_palette("viridis", len(sales_by_productline)),  # Color palette for the labels
              alpha=0.7)  # Set transparency for the squares

# Title and Display
plt.title('TreeMap of Sales by Product Line', fontsize=16)  # Add title
plt.axis('off')  # Turn off the axis for better visualization
plt.show()  # Display the tree map
