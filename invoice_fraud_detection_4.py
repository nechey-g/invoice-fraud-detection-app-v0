import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import networkx as nx
import matplotlib.pyplot as plt
from pyod.models.auto_encoder import AutoEncoder
import seaborn as sns
from st_aggrid import AgGrid, GridOptionsBuilder
import time
from streamlit.runtime.scriptrunner import add_script_run_ctx
import threading
import os
import ast

# Set page config first - this must be the first Streamlit command
st.set_page_config(
    page_title="Invoice Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Faker for realistic fake data
fake = Faker()

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# =============================================
# DATA LOADING FROM SAVED CSV FILES
# =============================================

def load_data():
    """Load invoice and taxpayer data from CSV files"""
    invoice_df = pd.read_csv("invoice_data.csv")
    taxpayer_df = pd.read_csv("taxpayer_data.csv")
    
    # Convert date columns to datetime
    invoice_df['invoice_date'] = pd.to_datetime(invoice_df['invoice_date'])
    invoice_df['due_date'] = pd.to_datetime(invoice_df['due_date'])
    taxpayer_df['registration_date'] = pd.to_datetime(taxpayer_df['registration_date'])
    
    # Convert line_items from string to list of dictionaries
    if isinstance(invoice_df['line_items'].iloc[0], str):
        import ast
        invoice_df['line_items'] = invoice_df['line_items'].apply(ast.literal_eval)
    
    return invoice_df, taxpayer_df

# =============================================
# REAL-TIME ALERT SYSTEM
# =============================================

class RealTimeAlertSystem:
    def __init__(self, invoice_df, taxpayer_df):
        self.invoice_df = invoice_df
        self.taxpayer_df = taxpayer_df
        self.alerts = []
        self.alert_lock = threading.Lock()
        self.running = False
        self.thread = None
        
    def start_monitoring(self):
        """Start the real-time monitoring thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_invoices)
            add_script_run_ctx(self.thread)
            self.thread.start()
            
    def stop_monitoring(self):
        """Stop the real-time monitoring thread"""
        self.running = False
        if self.thread:
            self.thread.join()
            
    def _monitor_invoices(self):
        """Background thread that monitors for new high-risk invoices"""
        last_processed = len(self.invoice_df)
        
        while self.running:
            # Check for new invoices
            current_count = len(self.invoice_df)
            if current_count > last_processed:
                new_invoices = self.invoice_df.iloc[last_processed:current_count]
                self._process_new_invoices(new_invoices)
                last_processed = current_count
                
            # Sleep for a short interval
            time.sleep(5)
    
    def _process_new_invoices(self, new_invoices):
        """Process new invoices and generate alerts if needed"""
        for _, invoice in new_invoices.iterrows():
            # Check for high-risk indicators
            alerts = self._check_invoice_risks(invoice)
            
            if alerts:
                with self.alert_lock:
                    self.alerts.extend(alerts)
    
    def _check_invoice_risks(self, invoice):
        """Check an invoice for various risk factors"""
        alerts = []
        
        # 1. Check for high amount
        if invoice['grand_total'] > self.invoice_df['grand_total'].quantile(0.95):
            alerts.append({
                'type': 'High Value',
                'invoice_number': invoice['invoice_number'],
                'seller': invoice['seller_name'],
                'buyer': invoice['buyer_name'],
                'amount': invoice['grand_total'],
                'timestamp': datetime.now(),
                'severity': 'High',
                'message': f"High value invoice ({invoice['grand_total']}) detected between {invoice['seller_name']} and {invoice['buyer_name']}"
            })
        
        # 2. Check for zero VAT
        if invoice['tax_amount'] == 0 and invoice['grand_total'] > 1000:
            alerts.append({
                'type': 'Zero VAT',
                'invoice_number': invoice['invoice_number'],
                'seller': invoice['seller_name'],
                'buyer': invoice['buyer_name'],
                'amount': invoice['grand_total'],
                'timestamp': datetime.now(),
                'severity': 'Medium',
                'message': f"Zero VAT invoice with high amount ({invoice['grand_total']}) detected"
            })
        
        # 3. Check for missing delivery
        if not invoice['has_delivery'] and invoice['grand_total'] > 500:
            alerts.append({
                'type': 'Missing Delivery',
                'invoice_number': invoice['invoice_number'],
                'seller': invoice['seller_name'],
                'buyer': invoice['buyer_name'],
                'amount': invoice['grand_total'],
                'timestamp': datetime.now(),
                'severity': 'Medium',
                'message': f"Invoice without delivery documentation detected (amount: {invoice['grand_total']})"
            })
        
        # 4. Check for sector mismatch
        sector_products = {
            'Retail': ['Electronics', 'Clothing', 'Furniture', 'Groceries'],
            'Manufacturing': ['Raw Materials', 'Machinery Parts', 'Packaging', 'Tools'],
            'IT Services': ['Software License', 'Cloud Services', 'Consulting', 'Maintenance'],
            'Construction': ['Cement', 'Steel', 'Electrical', 'Plumbing'],
            'Healthcare': ['Medicines', 'Equipment', 'Supplies', 'Services']
        }
        
        for item in invoice['line_items']:
            if item['product'] not in sector_products.get(invoice['seller_sector'], []):
                alerts.append({
                    'type': 'Sector Mismatch',
                    'invoice_number': invoice['invoice_number'],
                    'seller': invoice['seller_name'],
                    'buyer': invoice['buyer_name'],
                    'product': item['product'],
                    'timestamp': datetime.now(),
                    'severity': 'Low',
                    'message': f"Product {item['product']} not typical for seller's sector ({invoice['seller_sector']})"
                })
                break
        
        # 5. Check for blacklisted taxpayers
        if invoice['seller_tin'] in set(self.taxpayer_df[self.taxpayer_df['is_blacklisted']]['TIN']):
            alerts.append({
                'type': 'Blacklisted Seller',
                'invoice_number': invoice['invoice_number'],
                'seller': invoice['seller_name'],
                'buyer': invoice['buyer_name'],
                'timestamp': datetime.now(),
                'severity': 'High',
                'message': f"Invoice from blacklisted seller: {invoice['seller_name']}"
            })
        
        return alerts
    
    def get_alerts(self, max_alerts=10):
        """Get the most recent alerts"""
        with self.alert_lock:
            return sorted(self.alerts, key=lambda x: x['timestamp'], reverse=True)[:max_alerts]
    
    def clear_alerts(self):
        """Clear all alerts"""
        with self.alert_lock:
            self.alerts = []

# =============================================
# 2. DESCRIPTIVE ANALYTICS
# =============================================

def descriptive_analytics(invoice_df, taxpayer_df):
    """Perform all descriptive analytics tasks from the document"""
    
    st.header("📊 Descriptive Analytics")
    st.write("""
    <div class="card" style="margin-bottom: 2rem;">
        This section summarizes and visualizes past and current invoice data to reveal patterns and anomalies.
    </div>
    """, unsafe_allow_html=True)
    
    # Task D_2: Aggregate invoice data by seller, buyer, product, sector, region
    st.subheader("Aggregated Invoice Data")
    
    agg_options = st.multiselect(
        "Select aggregation dimensions:",
        ['seller_sector', 'buyer_sector', 'payment_terms', 'has_delivery'],
        default=['seller_sector', 'buyer_sector']
    )
    
    if agg_options:
        agg_df = invoice_df.groupby(agg_options).agg({
            'invoice_number': 'count',
            'grand_total': ['sum', 'mean']
        }).reset_index()
        agg_df.columns = agg_options + ['invoice_count', 'total_amount', 'avg_amount']
        
        st.dataframe(agg_df.sort_values('total_amount', ascending=False))
        
        # Visualization
        fig = px.bar(agg_df, x=agg_options[0], y='total_amount', 
                     color=agg_options[1] if len(agg_options) > 1 else None,
                     title="Total Invoice Amount by Selected Dimensions")
        st.plotly_chart(fig, use_container_width=True)
    
    # Task D_3: Calculate total and average invoice value per taxpayer
    st.subheader("Taxpayer Invoice Statistics")
    
    seller_stats = invoice_df.groupby('seller_tin').agg({
        'invoice_number': 'count',
        'grand_total': ['sum', 'mean']
    }).reset_index()
    seller_stats.columns = ['seller_tin', 'invoice_count', 'total_amount', 'avg_amount']
    seller_stats = seller_stats.merge(taxpayer_df[['TIN', 'name', 'sector']], 
                                    left_on='seller_tin', right_on='TIN')
    
    buyer_stats = invoice_df.groupby('buyer_tin').agg({
        'invoice_number': 'count',
        'grand_total': ['sum', 'mean']
    }).reset_index()
    buyer_stats.columns = ['buyer_tin', 'invoice_count', 'total_amount', 'avg_amount']
    buyer_stats = buyer_stats.merge(taxpayer_df[['TIN', 'name', 'sector']], 
                                  left_on='buyer_tin', right_on='TIN')
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h4 style="color: #3498db;">Top Sellers by Invoice Volume</h4>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(seller_stats.sort_values('invoice_count', ascending=False).head(10))
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4 style="color: #3498db;">Top Buyers by Invoice Volume</h4>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(buyer_stats.sort_values('invoice_count', ascending=False).head(10))
    
    # Task D_4: Identify invoices with missing fields
    st.subheader("Missing Data Analysis")
    
    missing_data = pd.DataFrame({
        'Field': ['PO Number', 'Contract Reference', 'Delivery Note', 'Shipping Details'],
        'Missing Count': [
            invoice_df['po_number'].isna().sum(),
            invoice_df['contract_ref'].isna().sum(),
            invoice_df['delivery_note'].isna().sum(),
            invoice_df['shipping_details'].isna().sum()
        ],
        'Missing %': [
            round(invoice_df['po_number'].isna().mean() * 100, 1),
            round(invoice_df['contract_ref'].isna().mean() * 100, 1),
            round(invoice_df['delivery_note'].isna().mean() * 100, 1),
            round(invoice_df['shipping_details'].isna().mean() * 100, 1)
        ]
    })
    
    st.dataframe(missing_data)
    fig = px.bar(missing_data, x='Field', y='Missing %', title='Percentage of Missing Values by Field',
                color='Field', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)
    
    # Task D_5: Count invoices with no linked logistics/shipping or POS data
    st.markdown(f"""
    <div class="card">
        <h4 style="color: #e74c3c;">Invoices without delivery confirmation:</h4>
        <p style="font-size: 1.2rem;">
            {invoice_df[~invoice_df['has_delivery']].shape[0]} ({round((~invoice_df['has_delivery']).mean() * 100, 1)}%)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Task D_6: Identify high-frequency invoice issuers
    st.subheader("High-Frequency Invoice Issuers")
    
    freq_threshold = st.slider("Select frequency threshold (invoices per month):", 1, 20, 5)
    
    # Calculate invoices per month per seller - convert Period to string immediately
    invoice_df['invoice_month'] = pd.to_datetime(invoice_df['invoice_date']).dt.to_period('M').astype(str)
    freq_sellers = invoice_df.groupby(['seller_tin', 'invoice_month']).size().reset_index(name='count')
    high_freq_sellers = freq_sellers[freq_sellers['count'] >= freq_threshold]
    
    if not high_freq_sellers.empty:
        high_freq_sellers = high_freq_sellers.merge(taxpayer_df[['TIN', 'name', 'sector']], 
                                                  left_on='seller_tin', right_on='TIN')
        st.dataframe(high_freq_sellers.sort_values('count', ascending=False))
        
        # Visualize
        fig = px.line(high_freq_sellers, x='invoice_month', y='count', color='name',
                     title=f"Sellers with ≥{freq_threshold} invoices/month",
                     line_shape='spline', render_mode='svg')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write(f"No sellers issued ≥{freq_threshold} invoices in any month")
    
    # Task D_7: Create monthly/quarterly invoice volume/value trends
    st.subheader("Invoice Trends Over Time")
    
    time_group = st.radio("Time period:", ['Monthly', 'Quarterly'])
    
    if time_group == 'Monthly':
        # Convert to string immediately to avoid Period serialization issues
        invoice_df['period'] = pd.to_datetime(invoice_df['invoice_date']).dt.to_period('M').astype(str)
    else:
        invoice_df['period'] = pd.to_datetime(invoice_df['invoice_date']).dt.to_period('Q').astype(str)
    
    trend_df = invoice_df.groupby('period').agg({
        'invoice_number': 'count',
        'grand_total': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df['period'], y=trend_df['invoice_number'],
                         name='Invoice Count', line=dict(color='#3498db', width=3)))
    fig.add_trace(go.Scatter(x=trend_df['period'], y=trend_df['grand_total']/1000,
                         name='Total Amount (Thousand)', yaxis='y2', line=dict(color='#2ecc71', width=3)))
    
    fig.update_layout(
        title='Invoice Volume and Value Over Time',
        yaxis=dict(title='Invoice Count'),
        yaxis2=dict(title='Total Amount (Thousand)', overlaying='y', side='right'),
        xaxis=dict(title='Period'),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Task D_8: Visualize top taxpayers by invoice volume/value
    st.subheader("Top Taxpayers")
    
    top_n = st.slider("Number of top taxpayers to show:", 5, 20, 10)
    metric = st.radio("Rank by:", ['Invoice Count', 'Total Amount'])
    
    if metric == 'Invoice Count':
        top_df = seller_stats.sort_values('invoice_count', ascending=False).head(top_n)
        fig = px.bar(top_df, x='name', y='invoice_count', color='sector',
                    title=f"Top {top_n} Sellers by Invoice Count",
                    color_discrete_sequence=px.colors.qualitative.Pastel)
    else:
        top_df = seller_stats.sort_values('total_amount', ascending=False).head(top_n)
        fig = px.bar(top_df, x='name', y='total_amount', color='sector',
                    title=f"Top {top_n} Sellers by Total Amount",
                    color_discrete_sequence=px.colors.qualitative.Pastel)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Task D_9: Summarize invoice patterns by tax type and taxpayer category
    st.subheader("Tax Patterns by Sector")
    
    tax_df = invoice_df.groupby('seller_sector').agg({
        'tax_rate': 'mean',
        'tax_amount': 'sum',
        'invoice_number': 'count'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(tax_df, x='seller_sector', y='tax_rate', 
                    title='Average Tax Rate by Sector',
                    color='seller_sector',
                    color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(tax_df, x='seller_sector', y='tax_amount', 
                    title='Total Tax Amount by Sector',
                    color='seller_sector',
                    color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    
    # Task D_10: Build dashboards for suspicious invoice indicators
    st.subheader("Suspicious Invoice Indicators")
    
    indicators = {
        'Zero VAT': (invoice_df['tax_amount'] == 0) & (invoice_df['grand_total'] > 1000),
        'No Delivery': ~invoice_df['has_delivery'],
        'Round Amounts': invoice_df['grand_total'].apply(lambda x: x.is_integer()),
        'High Value': invoice_df['grand_total'] > invoice_df['grand_total'].quantile(0.99)
    }
    
    indicator_df = pd.DataFrame({
        'Indicator': indicators.keys(),
        'Count': [sum(ind) for ind in indicators.values()],
        'Percentage': [round(sum(ind)/len(invoice_df)*100, 1) for ind in indicators.values()]
    })
    
    st.dataframe(indicator_df)
    
    fig = px.bar(indicator_df, x='Indicator', y='Count', color='Indicator',
                title='Count of Suspicious Invoice Indicators',
                color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)

# =============================================
# 3. DIAGNOSTIC ANALYTICS
# =============================================

def diagnostic_analytics(invoice_df, taxpayer_df):
    """Perform all diagnostic analytics tasks from the document"""
    
    st.header("🔍 Diagnostic Analytics")
    st.write("""
    <div class="card" style="margin-bottom: 2rem;">
        This section helps understand the root causes of anomalies or suspicious invoices.
    </div>
    """, unsafe_allow_html=True)
    
    # Task DA_1: Match seller's TP_CATEGORY with product categories in invoices
    st.subheader("Sector-Product Mismatch Analysis")
    
    # Create a list of expected products for each sector
    sector_products = {
        'Retail': ['Electronics', 'Clothing', 'Furniture', 'Groceries'],
        'Manufacturing': ['Raw Materials', 'Machinery Parts', 'Packaging', 'Tools'],
        'IT Services': ['Software License', 'Cloud Services', 'Consulting', 'Maintenance'],
        'Construction': ['Cement', 'Steel', 'Electrical', 'Plumbing'],
        'Healthcare': ['Medicines', 'Equipment', 'Supplies', 'Services']
    }
    
    # Check for each line item if product matches seller's sector
    mismatch_invoices = []
    for _, row in invoice_df.iterrows():
        for item in row['line_items']:
            if item['product'] not in sector_products.get(row['seller_sector'], []):
                mismatch_invoices.append(row['invoice_number'])
                break  # Only count once per invoice
    
    st.markdown(f"""
    <div class="card">
        <h4 style="color: #e74c3c;">Invoices with products outside seller's registered sector:</h4>
        <p style="font-size: 1.2rem;">
            {len(mismatch_invoices)} ({round(len(mismatch_invoices)/len(invoice_df)*100, 1)}%)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(mismatch_invoices) > 0:
        mismatch_df = invoice_df[invoice_df['invoice_number'].isin(mismatch_invoices)]
        st.dataframe(mismatch_df[['invoice_number', 'seller_name', 'seller_sector', 'line_items']].head(10))
    
    # Task DA_2: Validate seller and buyer TINs against taxpayer registry
    st.subheader("Taxpayer Validation")
    
    all_tins = set(taxpayer_df['TIN'])
    invalid_sellers = set(invoice_df['seller_tin']) - all_tins
    invalid_buyers = set(invoice_df['buyer_tin']) - all_tins
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="card">
            <h4 style="color: #e74c3c;">Invoices with unregistered sellers:</h4>
            <p style="font-size: 1.2rem;">
                {len(invoice_df[invoice_df['seller_tin'].isin(invalid_sellers)])}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="card">
            <h4 style="color: #e74c3c;">Invoices with unregistered buyers:</h4>
            <p style="font-size: 1.2rem;">
                {len(invoice_df[invoice_df['buyer_tin'].isin(invalid_buyers)])}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Task DA_3: Cross-match invoices with logistics/delivery data to find phantom invoices
    st.subheader("Potential Phantom Invoices")
    
    phantom_df = invoice_df[(~invoice_df['has_delivery']) & 
                          (invoice_df['grand_total'] > 1000)]
    
    st.markdown(f"""
    <div class="card">
        <h4 style="color: #e74c3c;">Potential phantom invoices (no delivery + high value):</h4>
        <p style="font-size: 1.2rem;">
            {len(phantom_df)}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(phantom_df) > 0:
        st.dataframe(phantom_df[['invoice_number', 'seller_name', 'buyer_name', 'grand_total']].sort_values('grand_total', ascending=False).head(10))
    
    # Task DA_4: Analyze duplicate invoice numbers or entries with minor variations
    st.subheader("Duplicate Invoice Analysis")
    
    # Check for duplicate invoice numbers
    dup_numbers = invoice_df['invoice_number'].value_counts()
    dup_numbers = dup_numbers[dup_numbers > 1].index.tolist()
    
    st.markdown(f"""
    <div class="card">
        <h4 style="color: #e74c3c;">Duplicate invoice numbers found:</h4>
        <p style="font-size: 1.2rem;">
            {len(dup_numbers)}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(dup_numbers) > 0:
        dup_df = invoice_df[invoice_df['invoice_number'].isin(dup_numbers)]
        st.dataframe(dup_df.sort_values('invoice_number').head(10))
    
    # Task DA_5: Detect circular transactions between seller and buyer pairs using graph models
    st.subheader("Circular Transaction Detection")
    
    # Create a directed graph of transactions
    G = nx.DiGraph()
    
    # Add edges with weights based on transaction frequency
    transaction_counts = invoice_df.groupby(['seller_tin', 'buyer_tin']).size().reset_index(name='count')
    
    for _, row in transaction_counts.iterrows():
        G.add_edge(row['seller_tin'], row['buyer_tin'], weight=row['count'])
    
    # Find strongly connected components (potential circular transactions)
    scc = list(nx.strongly_connected_components(G))
    circular_pairs = [comp for comp in scc if len(comp) > 1]
    
    st.markdown(f"""
    <div class="card">
        <h4 style="color: #e74c3c;">Potential circular transaction groups detected:</h4>
        <p style="font-size: 1.2rem;">
            {len(circular_pairs)}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(circular_pairs) > 0:
        for i, group in enumerate(circular_pairs[:3]):  # Show top 3 groups
            st.write(f"Group {i+1}:")
            members = taxpayer_df[taxpayer_df['TIN'].isin(group)][['TIN', 'name', 'sector']]
            st.dataframe(members)
            
            # Visualize the subgraph
            subgraph = G.subgraph(group)
            fig, ax = plt.subplots(figsize=(8, 6))
            pos = nx.spring_layout(subgraph)
            nx.draw(subgraph, pos, with_labels=True, 
                   labels={n: taxpayer_df[taxpayer_df['TIN'] == n]['name'].values[0] for n in subgraph.nodes()},
                   node_size=2000, node_color='lightblue', 
                   edge_color='gray', width=[d['weight']*0.5 for u, v, d in subgraph.edges(data=True)],
                   ax=ax)
            plt.title(f"Circular Transaction Group {i+1}")
            st.pyplot(fig)
    
    # Task DA_6: Correlate invoice anomalies with taxpayer audit/reassessment history
    st.subheader("Anomaly Correlation with Blacklisted Taxpayers")
    
    blacklisted_tins = set(taxpayer_df[taxpayer_df['is_blacklisted']]['TIN'])
    
    # Calculate anomaly rates for blacklisted vs non-blacklisted taxpayers
    blacklisted_sales = invoice_df[invoice_df['seller_tin'].isin(blacklisted_tins)]
    normal_sales = invoice_df[~invoice_df['seller_tin'].isin(blacklisted_tins)]
    
    anomaly_metrics = pd.DataFrame({
        'Metric': ['Avg Invoice Amount', 'Zero VAT %', 'No Delivery %', 'Sector Mismatch %'],
        'Blacklisted': [
            blacklisted_sales['grand_total'].mean(),
            (blacklisted_sales['tax_amount'] == 0).mean() * 100,
            (~blacklisted_sales['has_delivery']).mean() * 100,
            len([inv for inv in blacklisted_sales['invoice_number'] if inv in mismatch_invoices]) / len(blacklisted_sales) * 100
        ],
        'Non-Blacklisted': [
            normal_sales['grand_total'].mean(),
            (normal_sales['tax_amount'] == 0).mean() * 100,
            (~normal_sales['has_delivery']).mean() * 100,
            len([inv for inv in normal_sales['invoice_number'] if inv in mismatch_invoices]) / len(normal_sales) * 100
        ]
    })
    
    st.dataframe(anomaly_metrics)
    
    fig = px.bar(anomaly_metrics, x='Metric', y=['Blacklisted', 'Non-Blacklisted'],
                barmode='group', title='Anomaly Comparison: Blacklisted vs Non-Blacklisted Sellers',
                color_discrete_sequence=['#e74c3c', '#3498db'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Task DA_7: Identify transactions that violate VAT rules
    st.subheader("VAT Rule Violations")
    
    # Check for invoices with VAT but shouldn't have (based on sector)
    vat_violations = invoice_df[
        ((invoice_df['seller_sector'] == 'Healthcare') & (invoice_df['tax_amount'] > 0)) |
        ((invoice_df['seller_sector'] == 'IT Services') & (invoice_df['tax_rate'] > 0.1))  # Assuming max 10% for IT
    ]
    
    st.markdown(f"""
    <div class="card">
        <h4 style="color: #e74c3c;">Potential VAT rule violations:</h4>
        <p style="font-size: 1.2rem;">
            {len(vat_violations)}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(vat_violations) > 0:
        st.dataframe(vat_violations[['invoice_number', 'seller_name', 'seller_sector', 'tax_rate', 'tax_amount']].head(10))
    
    # Task DA_8: Compare invoice prices/quantities with sector/industry benchmarks
    st.subheader("Price/Quantity Anomalies")
    
    # Calculate sector benchmarks
    sector_benchmarks = invoice_df.groupby('seller_sector').agg({
        'grand_total': ['mean', 'std'],
    }).reset_index()
    sector_benchmarks.columns = ['sector', 'avg_amount', 'std_amount']
    
    # Identify outliers (2+ std from mean)
    invoice_df = invoice_df.merge(sector_benchmarks, left_on='seller_sector', right_on='sector')
    invoice_df['amount_zscore'] = (invoice_df['grand_total'] - invoice_df['avg_amount']) / invoice_df['std_amount']
    price_outliers = invoice_df[abs(invoice_df['amount_zscore']) > 2]
    
    st.markdown(f"""
    <div class="card">
        <h4 style="color: #e74c3c;">Invoices with unusually high/low amounts for their sector:</h4>
        <p style="font-size: 1.2rem;">
            {len(price_outliers)}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(price_outliers) > 0:
        st.dataframe(price_outliers[['invoice_number', 'seller_name', 'seller_sector', 'grand_total', 'avg_amount', 'amount_zscore']].sort_values('amount_zscore', key=abs, ascending=False).head(10))
    
    # Task DA_9: Examine seasonality effects on suspicious invoice spikes
    st.subheader("Seasonality of Suspicious Invoices")
    
    # Convert to string immediately to avoid Period serialization issues
    invoice_df['invoice_month'] = pd.to_datetime(invoice_df['invoice_date']).dt.to_period('M').astype(str)
    fraud_monthly = invoice_df.groupby('invoice_month')['is_fraud'].agg(['count', 'sum']).reset_index()
    fraud_monthly['fraud_rate'] = fraud_monthly['sum'] / fraud_monthly['count'] * 100
    
    fig = px.line(fraud_monthly, x='invoice_month', y='fraud_rate',
                 title='Monthly Fraud Rate Trend',
                 line_shape='spline', render_mode='svg')
    st.plotly_chart(fig, use_container_width=True)
    
    # Task DA_10: Trace taxpayers with sudden surges in invoice activity or volume
    st.subheader("Sudden Activity Spikes")
    
    # Calculate monthly invoice counts per seller
    seller_monthly = invoice_df.groupby(['seller_tin', 'invoice_month']).size().reset_index(name='count')
    
    # Calculate z-scores for each seller's monthly activity
    seller_stats = seller_monthly.groupby('seller_tin')['count'].agg(['mean', 'std']).reset_index()
    seller_monthly = seller_monthly.merge(seller_stats, on='seller_tin')
    seller_monthly['zscore'] = (seller_monthly['count'] - seller_monthly['mean']) / seller_monthly['std']
    
    # Identify spikes (z-score > 3)
    spikes = seller_monthly[seller_monthly['zscore'] > 3].sort_values('zscore', ascending=False)
    spikes = spikes.merge(taxpayer_df[['TIN', 'name', 'sector']], left_on='seller_tin', right_on='TIN')
    
    st.markdown(f"""
    <div class="card">
        <h4 style="color: #e74c3c;">Sellers with sudden activity spikes:</h4>
        <p style="font-size: 1.2rem;">
            {len(spikes)}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(spikes) > 0:
        st.dataframe(spikes[['name', 'sector', 'invoice_month', 'count', 'mean', 'zscore']].head(10))
        
        # Plot example spike
        example_tin = spikes.iloc[0]['seller_tin']
        example_data = seller_monthly[seller_monthly['seller_tin'] == example_tin]
        
        fig = px.line(example_data, x='invoice_month', y='count',
                     title=f"Monthly Invoice Count for {spikes.iloc[0]['name']}",
                     line_shape='spline', render_mode='svg')
        fig.add_hline(y=example_data['mean'].iloc[0], line_dash="dash", 
                     annotation_text="Mean", annotation_position="bottom right")
        st.plotly_chart(fig, use_container_width=True)

# =============================================
# 4. PREDICTIVE ANALYTICS
# =============================================

def predictive_analytics(invoice_df, taxpayer_df):
    """Perform all predictive analytics tasks from the document"""
    
    st.header("🔮 Predictive Analytics")
    st.write("""
    <div class="card" style="margin-bottom: 2rem;">
        This section uses machine learning to predict whether invoices are likely fraudulent.
    </div>
    """, unsafe_allow_html=True)
    
    # Task P_1: Create labeled dataset (we already have is_fraud in our simulated data)
    
    # Task P_2: Engineer features
    st.subheader("Feature Engineering")
    
    # Create features for each invoice
    features = []
    
    for _, row in invoice_df.iterrows():
        # Basic features
        invoice_features = {
            'invoice_number': row['invoice_number'],
            'seller_tin': row['seller_tin'],
            'amount': row['grand_total'],
            'tax_rate': row['tax_rate'],
            'tax_amount': row['tax_amount'],
            'has_delivery': row['has_delivery'],
            'line_items_count': len(row['line_items']),
            'avg_line_amount': sum(item['line_total'] for item in row['line_items']) / len(row['line_items']) if row['line_items'] else 0,
            'is_round_amount': row['grand_total'].is_integer(),
            'zero_vat': row['tax_amount'] == 0,
            'po_exists': not pd.isna(row['po_number']),
            'contract_exists': not pd.isna(row['contract_ref']),
            'seller_sector': row['seller_sector'],
            'buyer_sector': row['buyer_sector'],
            'sector_mismatch': any(item['product'] not in {
                'Retail': ['Electronics', 'Clothing', 'Furniture', 'Groceries'],
                'Manufacturing': ['Raw Materials', 'Machinery Parts', 'Packaging', 'Tools'],
                'IT Services': ['Software License', 'Cloud Services', 'Consulting', 'Maintenance'],
                'Construction': ['Cement', 'Steel', 'Electrical', 'Plumbing'],
                'Healthcare': ['Medicines', 'Equipment', 'Supplies', 'Services']
            }.get(row['seller_sector'], []) for item in row['line_items']),
            'is_fraud': row['is_fraud']
        }
        
        # Add seller historical features
        seller_history = invoice_df[invoice_df['seller_tin'] == row['seller_tin']]
        invoice_features.update({
            'seller_invoice_count': len(seller_history),
            'seller_avg_amount': seller_history['grand_total'].mean(),
            'seller_fraud_rate': seller_history['is_fraud'].mean(),
            'seller_avg_line_items': seller_history['line_items'].apply(len).mean()
        })
        
        # Add buyer historical features
        buyer_history = invoice_df[invoice_df['buyer_tin'] == row['buyer_tin']]
        invoice_features.update({
            'buyer_invoice_count': len(buyer_history),
            'buyer_avg_amount': buyer_history['grand_total'].mean(),
            'buyer_fraud_rate': buyer_history['is_fraud'].mean()
        })
        
        features.append(invoice_features)
    
    features_df = pd.DataFrame(features)
    
    # One-hot encode categorical variables
    features_df = pd.get_dummies(features_df, columns=['seller_sector', 'buyer_sector'])
    
    st.write("Sample of engineered features:")
    st.dataframe(features_df.head())
    
    # Task P_3: Train classification models
    st.subheader("Model Training")
    
    # Prepare data
    X = features_df.drop(columns=['invoice_number', 'seller_tin', 'is_fraud'])
    y = features_df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    model_results = []
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics - handle case where there might be no positive cases in test set
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Safely get metrics (in case there are no fraud cases in the test set)
        precision = report['1']['precision'] if '1' in report else 0
        recall = report['1']['recall'] if '1' in report else 0
        f1 = report['1']['f1-score'] if '1' in report else 0
        
        auc_score = roc_auc_score(y_test, y_prob)
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall_vals, precision_vals)
        
        model_results.append({
            'Model': name,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC-ROC': auc_score,
            'AUC-PR': pr_auc
        })
        
        # Plot feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = px.bar(importance, x='Importance', y='Feature', orientation='h',
                         title=f'{name} - Top Feature Importances',
                         color='Importance',
                         color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
    
    # Display model comparison
    st.subheader("Model Comparison")
    results_df = pd.DataFrame(model_results)
    st.dataframe(results_df.sort_values('AUC-ROC', ascending=False))
    
    # Plot ROC curves
    fig = go.Figure()
    
    for name, model in models.items():
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'{name} (AUC = {auc_score:.2f})',
            mode='lines',
            line=dict(width=3)
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random',
        line=dict(dash='dash', color='grey')
    ))
    
    fig.update_layout(
        title='ROC Curves',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Task P_4: Train unsupervised anomaly models
    st.subheader("Unsupervised Anomaly Detection")
    
    # Since we have labels, we'll just demonstrate one unsupervised approach
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_scores = iso_forest.fit_predict(X_train_scaled)
    
    # Convert scores to binary predictions (1 for inlier, -1 for outlier)
    iso_preds = np.where(iso_scores == -1, 1, 0)
    
    # Evaluate against actual fraud labels (just for demonstration)
    iso_report = classification_report(y_train, iso_preds, output_dict=True)
    
    st.write("Isolation Forest Results (on training data):")
    st.json({
        'Precision': iso_report['1']['precision'] if '1' in iso_report else 0,
        'Recall': iso_report['1']['recall'] if '1' in iso_report else 0,
        'F1-Score': iso_report['1']['f1-score'] if '1' in iso_report else 0
    })
    
    # Task P_6: Generate fraud risk scores for each invoice or taxpayer
    st.subheader("Fraud Risk Scoring")
    
    # Use the best model (Random Forest in this case) to score all invoices
    best_model = models['Random Forest']
    features_df['fraud_risk'] = best_model.predict_proba(scaler.transform(X))[:, 1]
    
    # Show high-risk invoices
    st.write("Invoices with highest fraud risk:")
    high_risk = features_df.sort_values('fraud_risk', ascending=False).head(10)
    st.dataframe(high_risk[['invoice_number', 'fraud_risk', 'is_fraud']])
    
    # Task P_7: Detect clusters of high-risk taxpayers
    st.subheader("Taxpayer Risk Clustering")
    
    # Aggregate by seller
    seller_risk = features_df.groupby('seller_tin').agg({
        'fraud_risk': 'mean',
        'amount': 'sum',
        'line_items_count': 'sum'
    }).reset_index()
    
    seller_risk = seller_risk.merge(taxpayer_df[['TIN', 'name', 'sector']], left_on='seller_tin', right_on='TIN')
    
    # Scale features for clustering
    cluster_features = seller_risk[['fraud_risk', 'amount', 'line_items_count']]
    cluster_scaler = StandardScaler()
    cluster_scaled = cluster_scaler.fit_transform(cluster_features)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    seller_risk['cluster'] = kmeans.fit_predict(cluster_scaled)
    
    # Visualize clusters
    fig = px.scatter_3d(seller_risk, x='fraud_risk', y='amount', z='line_items_count',
                       color='cluster', hover_name='name', hover_data=['sector'],
                       title='Taxpayer Risk Clusters',
                       color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    # Show high-risk cluster - FIXED SECTION
    numeric_cols = seller_risk.select_dtypes(include=[np.number]).columns
    cluster_means = seller_risk.groupby('cluster')[numeric_cols].mean()
    high_risk_cluster = cluster_means['fraud_risk'].idxmax()
    
    st.write(f"High-Risk Cluster {high_risk_cluster} Taxpayers:")
    st.dataframe(seller_risk[seller_risk['cluster'] == high_risk_cluster].sort_values('fraud_risk', ascending=False).head(10))
    
    return features_df  # Return features with risk scores for prescriptive analytics

# =============================================
# 5. PRESCRIPTIVE ANALYTICS
# =============================================

def prescriptive_analytics(invoice_df, taxpayer_df, features_df):
    """Perform all prescriptive analytics tasks from the document"""
    
    st.header("🎯 Prescriptive Analytics")
    st.write("""
    <div class="card" style="margin-bottom: 2rem;">
        This section recommends actions based on the predictive models and diagnostic insights.
    </div>
    """, unsafe_allow_html=True)
    
    # Merge fraud risk scores with original invoice data
    invoice_df = invoice_df.merge(features_df[['invoice_number', 'fraud_risk']], on='invoice_number')
    
    # Task PR_1: Design rule-based thresholds to trigger fraud investigations
    st.subheader("Fraud Investigation Thresholds")
    
    risk_threshold = st.slider("Set fraud risk threshold for investigation:", 0.1, 1.0, 0.9, 0.05)
    
    flagged_invoices = invoice_df[invoice_df['fraud_risk'] >= risk_threshold]
    st.markdown(f"""
    <div class="card">
        <h4 style="color: #e74c3c;">Invoices flagged for investigation (risk ≥ {risk_threshold}):</h4>
        <p style="font-size: 1.2rem;">
            {len(flagged_invoices)}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(flagged_invoices) > 0:
        st.dataframe(flagged_invoices[['invoice_number', 'seller_name', 'buyer_name', 'grand_total', 'fraud_risk', 'is_fraud']].sort_values('fraud_risk', ascending=False).head(10))
    
    # Task PR_2: Prioritize taxpayers for audit based on risk scores and volume
    st.subheader("Taxpayer Audit Prioritization")
    
    # Aggregate by seller
    seller_audit = invoice_df.groupby(['seller_tin', 'seller_name']).agg({
        'invoice_number': 'count',
        'grand_total': 'sum',
        'fraud_risk': 'mean',
        'is_fraud': 'sum'
    }).reset_index()
    
    seller_audit['audit_priority'] = seller_audit['fraud_risk'] * seller_audit['grand_total']
    seller_audit = seller_audit.sort_values('audit_priority', ascending=False)
    
    st.write("Top taxpayers recommended for audit:")
    st.dataframe(seller_audit.head(10))
    
    # Visualize audit priorities
    fig = px.scatter(seller_audit.head(20), x='invoice_number', y='grand_total',
                    size='fraud_risk', color='fraud_risk',
                    hover_name='seller_name', log_x=True, log_y=True,
                    title='Audit Priority: Size = Fraud Risk, Color = Fraud Risk',
                    color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    # Task PR_3: Recommend sector-level policies
    st.subheader("Sector-Level Policy Recommendations")
    
    sector_risk = invoice_df.groupby('seller_sector').agg({
        'fraud_risk': 'mean',
        'is_fraud': 'mean'
    }).reset_index().sort_values('fraud_risk', ascending=False)
    
    st.write("Highest risk sectors:")
    st.dataframe(sector_risk)
    
    high_risk_sector = sector_risk.iloc[0]['seller_sector']
    st.markdown(f"""
    <div class="card">
        <h4 style="color: #e74c3c;">Recommendation for {high_risk_sector} sector:</h4>
        <ul style="font-size: 1rem;">
            <li>Implement stricter invoice validation requirements</li>
            <li>Require additional documentation for high-value transactions</li>
            <li>Conduct targeted awareness programs</li>
            <li>Increase frequency of random audits</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Task PR_4: Simulate outcomes of different audit strategies
    st.subheader("Audit Strategy Simulation")
    
    strategy = st.radio("Select audit strategy:", 
                       ['Random Sampling', 'Risk-Based', 'Sector-Focused', 'High-Value Focus'])
    
    sample_size = st.slider("Sample size for simulation:", 10, 200, 50)
    
    if strategy == 'Random Sampling':
        sample = invoice_df.sample(sample_size)
    elif strategy == 'Risk-Based':
        sample = invoice_df.sort_values('fraud_risk', ascending=False).head(sample_size)
    elif strategy == 'Sector-Focused':
        target_sector = sector_risk.iloc[0]['seller_sector']
        sector_invoices = invoice_df[invoice_df['seller_sector'] == target_sector]
        sample = sector_invoices.sort_values('fraud_risk', ascending=False).head(sample_size)
    else:  # High-Value Focus
        sample = invoice_df.sort_values('grand_total', ascending=False).head(sample_size)
    
    fraud_detection_rate = sample['is_fraud'].mean() * 100
    avg_fraud_amount = sample[sample['is_fraud']]['grand_total'].mean()
    
    st.markdown(f"""
    <div class="card">
        <h4 style="color: #2ecc71;">Simulation Results:</h4>
        <ul style="font-size: 1rem;">
            <li>Fraud detection rate: {fraud_detection_rate:.1f}%</li>
            <li>Average fraud amount detected: ${avg_fraud_amount:,.2f}</li>
            <li>Total potential revenue recovered: ${sample[sample['is_fraud']]['grand_total'].sum():,.2f}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Task PR_5: Automate alerts for invoices breaching business rules
    st.subheader("Automated Alert Rules")
    
    alert_rules = {
        'High Risk Score': invoice_df['fraud_risk'] > risk_threshold,
        'No Delivery + High Value': (~invoice_df['has_delivery']) & (invoice_df['grand_total'] > 1000),
        'Sector Mismatch': [any(item['product'] not in {
            'Retail': ['Electronics', 'Clothing', 'Furniture', 'Groceries'],
            'Manufacturing': ['Raw Materials', 'Machinery Parts', 'Packaging', 'Tools'],
            'IT Services': ['Software License', 'Cloud Services', 'Consulting', 'Maintenance'],
            'Construction': ['Cement', 'Steel', 'Electrical', 'Plumbing'],
            'Healthcare': ['Medicines', 'Equipment', 'Supplies', 'Services']
        }.get(row['seller_sector'], []) for item in row['line_items']) for _, row in invoice_df.iterrows()],
        'Zero VAT + High Value': (invoice_df['tax_amount'] == 0) & (invoice_df['grand_total'] > 1000)
    }
    
    alert_counts = {rule: sum(condition) for rule, condition in alert_rules.items()}
    
    st.write("Potential alert rules and their frequencies:")
    st.dataframe(pd.DataFrame.from_dict(alert_counts, orient='index', columns=['Count']))
    
    # Task PR_6: Suggest real-time blocking of invoices with missing/invalid fields
    st.subheader("Data Validation Rules")
    
    validation_rules = {
        'Missing TIN': (invoice_df['seller_tin'].isna()) | (invoice_df['buyer_tin'].isna()),
        'Invalid TIN': (~invoice_df['seller_tin'].isin(taxpayer_df['TIN'])) | (~invoice_df['buyer_tin'].isin(taxpayer_df['TIN'])),
        'Missing Product Info': invoice_df['line_items'].apply(lambda x: any('product' not in item or not item['product'] for item in x)),
        'Negative Amounts': invoice_df['grand_total'] < 0
    }
    
    validation_counts = {rule: sum(condition) for rule, condition in validation_rules.items()}
    
    st.write("Data validation issues that could trigger blocking:")
    st.dataframe(pd.DataFrame.from_dict(validation_counts, orient='index', columns=['Count']))
    
    st.subheader("Summary Recommendations")
    st.markdown("""
    <div class="card">
        <h4 style="color: #3498db;">1. Immediate Actions:</h4>
        <ul style="font-size: 1rem;">
            <li>Audit top 20 high-risk taxpayers identified</li>
            <li>Block invoices with missing critical fields (TIN, product info)</li>
            <li>Implement real-time alerts for high-risk invoices</li>
        </ul>
        
        <h4 style="color: #3498db; margin-top: 1rem;">2. Short-Term (Next 3 Months):</h4>
        <ul style="font-size: 1rem;">
            <li>Conduct targeted audits in high-risk sectors</li>
            <li>Implement stricter validation for zero-VAT invoices</li>
            <li>Develop taxpayer education program for high-risk sectors</li>
        </ul>
        
        <h4 style="color: #3498db; margin-top: 1rem;">3. Long-Term (Next 6-12 Months):</h4>
        <ul style="font-size: 1rem;">
            <li>Implement continuous monitoring system with ML models</li>
            <li>Develop compliance rating system for taxpayers</li>
            <li>Optimize audit resource allocation based on predictive risk</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =============================================
# REAL-TIME ALERTS DASHBOARD
# =============================================

def real_time_alerts_dashboard(alert_system):
    """Dashboard for displaying real-time alerts"""
    
    st.header("🚨 Real-Time Alerts Dashboard")
    st.write("""
    <div class="card" style="margin-bottom: 2rem;">
        This dashboard shows real-time alerts for suspicious invoice activities.
    </div>
    """, unsafe_allow_html=True)
    
    # Alert controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Monitoring", key="start_monitoring"):
            alert_system.start_monitoring()
            st.success("Real-time monitoring started")
    
    with col2:
        if st.button("Stop Monitoring", key="stop_monitoring"):
            alert_system.stop_monitoring()
            st.warning("Real-time monitoring stopped")
    
    # Display current alerts
    st.subheader("Active Alerts")
    
    alerts = alert_system.get_alerts()
    
    if not alerts:
        st.info("No active alerts at this time")
    else:
        # Convert alerts to DataFrame for better display
        alert_df = pd.DataFrame(alerts)
        
        # Color coding based on severity
        def color_severity(severity):
            if severity == 'High':
                return 'color: #e74c3c; font-weight: bold;'
            elif severity == 'Medium':
                return 'color: #f39c12; font-weight: bold;'
            else:
                return 'color: #f1c40f;'
        
        # Apply styling
        styled_df = alert_df.style.applymap(lambda x: color_severity(x) if x in ['High', 'Medium', 'Low'] else '', subset=['severity'])
        
        # Display with AgGrid for better interactivity
        gb = GridOptionsBuilder.from_dataframe(alert_df)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=False)
        
        gridOptions = gb.build()
        AgGrid(alert_df, gridOptions=gridOptions, theme='streamlit', height=400)
        
        # Alert statistics
        st.subheader("Alert Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Alerts", len(alerts))
        with col2:
            high_alerts = len([a for a in alerts if a['severity'] == 'High'])
            st.metric("High Severity", high_alerts, delta=f"{high_alerts/len(alerts)*100:.1f}%")
        with col3:
            recent_alert = max([a['timestamp'] for a in alerts]) if alerts else "None"
            st.metric("Most Recent", recent_alert)
        
        # Visualize alert types
        alert_types = pd.DataFrame(alerts)['type'].value_counts().reset_index()
        alert_types.columns = ['Alert Type', 'Count']
        
        fig = px.bar(alert_types, x='Alert Type', y='Count', color='Alert Type',
                    title='Alert Types Distribution',
                    color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
        
        # Clear alerts button
        if st.button("Clear All Alerts"):
            alert_system.clear_alerts()
            st.experimental_rerun()

# =============================================
# MAIN APP WITH STREAMLIT DASHBOARD
# =============================================

def main():
    # Load data (cached for performance)
    @st.cache_data
    def cached_load_data():
        with st.spinner('🚀 Loading invoice data from CSV files...'):
            return load_data()
    
    invoice_df, taxpayer_df = cached_load_data()

    # Custom CSS for colorful animated styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        animation: gradientBG 15s ease infinite;
        background-size: 400% 400%;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        margin: 1rem;
        padding: 2rem;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(195deg, #2c3e50 0%, #1a1a2e 100%) !important;
        color: white;
        box-shadow: 5px 0 15px rgba(0,0,0,0.1);
    }
    
    .stHeader {
        color: #2c3e50;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(255,107,107,0.3);
        transition: all 0.4s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(255,107,107,0.4);
    }
    
    .stSelectbox, .stSlider, .stRadio {
        margin-bottom: 1.5rem;
        background-color: rgba(255,255,255,0.9);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .stSelectbox:hover, .stSlider:hover, .stRadio:hover {
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    .stDataFrame {
        border-radius: 12px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stDataFrame:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    }
    
    /* Animated cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border-left: 5px solid #3498db;
    }
    
    .card:hover {
        transform: translateY(-5px) scale(1.01);
        box-shadow: 0 15px 30px rgba(0,0,0,0.15);
    }
    
    /* Alert styling */
    .alert-high {
        border-left: 5px solid #e74c3c !important;
        animation: pulse 1.5s infinite;
    }
    
    .alert-medium {
        border-left: 5px solid #f39c12 !important;
    }
    
    .alert-low {
        border-left: 5px solid #f1c40f !important;
    }
    
    /* Pulse animation for important elements */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Floating animation */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .float {
        animation: float 6s ease-in-out infinite;
    }
    
        /* Custom scrollbar */
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #FF6B6B, #FF8E53);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #FF8E53, #FF6B6B);
    }
    
    /* Custom tooltip */
    [data-tooltip] {
        position: relative;
        cursor: pointer;
    }
    
    [data-tooltip]:hover:after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: #2c3e50;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.8rem;
        white-space: nowrap;
        z-index: 100;
    }
    
    /* Alert notification animation */
    @keyframes slideIn {
        from { transform: translateX(100%); }
        to { transform: translateX(0); }
    }
    
    .alert-notification {
        animation: slideIn 0.5s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize the alert system
    if 'alert_system' not in st.session_state:
        st.session_state.alert_system = RealTimeAlertSystem(invoice_df, taxpayer_df)
    
    # Animated header with floating elements
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("""
        <div style="text-align: center;">
            <svg width="100" height="100" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="float">
                <path d="M9 17V7M9 7L5 11M9 7L13 11M15 7V17M15 17L19 13M15 17L11 13" stroke="#3498db" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <h1 style="color: #2c3e50; margin-bottom: 0.5rem; font-size: 2.5rem;">
            <span style="color: #3498db;">Invoice</span> Fraud Detection System
        </h1>
        <p style="color: #7f8c8d; font-size: 1.1rem; margin-top: 0;">
            Interactive analytics dashboard with AI-powered detection
        </p>
        """, unsafe_allow_html=True)
    
    # Add a colorful animated divider
    st.markdown("""
    <div style="height: 3px; background: linear-gradient(90deg, #FF6B6B, #FF8E53, #3498db, #2ecc71); 
                margin: 1.5rem 0; border-radius: 3px; animation: gradientBG 8s ease infinite; 
                background-size: 300% 300%;"></div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation with icons and animations
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: white; display: flex; align-items: center; justify-content: center;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 10px;">
                    <path d="M3 12C3 4.5885 4.5885 3 12 3C19.4115 3 21 4.5885 21 12C21 19.4115 19.4115 21 12 21C4.5885 21 3 19.4115 3 12Z" stroke="white" stroke-width="2"/>
                    <path d="M16 12L8 12" stroke="white" stroke-width="2" stroke-linecap="round"/>
                    <path d="M12 16L12 8" stroke="white" stroke-width="2" stroke-linecap="round"/>
                </svg>
                Navigation
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        analysis_stage = st.radio(
            "Select Analysis Stage:",
            ["📊 Descriptive Analytics", 
             "🔍 Diagnostic Analytics", 
             "🔮 Predictive Analytics", 
             "🎯 Prescriptive Analytics",
             "🚨 Real-Time Alerts"],
            index=0
        )
        
        # Add some animated elements to sidebar
        st.markdown("""
        <div style="margin-top: 3rem; text-align: center;">
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 12px; 
                        border-left: 3px solid #3498db; animation: pulse 4s infinite;">
                <p style="color: white; margin: 0; font-size: 0.9rem;">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 5px; vertical-align: middle;">
                        <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="white" stroke-width="2"/>
                        <path d="M12 8V12" stroke="white" stroke-width="2" stroke-linecap="round"/>
                        <path d="M12 16H12.01" stroke="white" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                    <strong>Tip:</strong> Hover over charts for details
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display selected analysis stage with animated transitions
    if analysis_stage == "📊 Descriptive Analytics":
        with st.container():
            st.markdown("""
            <div class="card">
                <h3 style="color: #3498db; display: flex; align-items: center;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 10px;">
                        <path d="M3 3V21H21" stroke="#3498db" stroke-width="2" stroke-linecap="round"/>
                        <path d="M18 6L13 11L9 8L6 12" stroke="#3498db" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    Descriptive Analytics
                </h3>
                <p>Explore patterns and trends in invoice data</p>
            </div>
            """, unsafe_allow_html=True)
            descriptive_analytics(invoice_df, taxpayer_df)
            
    elif analysis_stage == "🔍 Diagnostic Analytics":
        with st.container():
            st.markdown("""
            <div class="card">
                <h3 style="color: #e74c3c; display: flex; align-items: center;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 10px;">
                        <path d="M15 7L20 12L15 17" stroke="#e74c3c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M4 12H19" stroke="#e74c3c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    Diagnostic Analytics
                </h3>
                <p>Investigate root causes of suspicious patterns</p>
            </div>
            """, unsafe_allow_html=True)
            diagnostic_analytics(invoice_df, taxpayer_df)
            
    elif analysis_stage == "🔮 Predictive Analytics":
        with st.container():
            st.markdown("""
            <div class="card">
                <h3 style="color: #9b59b6; display: flex; align-items: center;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 10px;">
                        <path d="M20 7L9 18L4 13" stroke="#9b59b6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    Predictive Analytics
                </h3>
                <p>AI models to detect potential fraud</p>
            </div>
            """, unsafe_allow_html=True)
            features_df = predictive_analytics(invoice_df, taxpayer_df)
            
    elif analysis_stage == "🎯 Prescriptive Analytics":
        with st.container():
            st.markdown("""
            <div class="card">
                <h3 style="color: #2ecc71; display: flex; align-items: center;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 10px;">
                        <path d="M5 13L9 17L19 7" stroke="#2ecc71" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    Prescriptive Analytics
                </h3>
                <p>Actionable recommendations based on findings</p>
            </div>
            """, unsafe_allow_html=True)
            if 'features_df' not in globals():
                features_df = predictive_analytics(invoice_df, taxpayer_df)
            prescriptive_analytics(invoice_df, taxpayer_df, features_df)
    
    elif analysis_stage == "🚨 Real-Time Alerts":
        with st.container():
            real_time_alerts_dashboard(st.session_state.alert_system)
    
    # Add animated footer with social links
    st.markdown("""
    <div style="margin-top: 5rem; text-align: center; padding: 1.5rem; background: rgba(52, 152, 219, 0.1); border-radius: 12px;">
        <div style="display: flex; justify-content: center; gap: 1.5rem; margin-bottom: 1rem;">
            <a href="#" style="color: #2c3e50; text-decoration: none; transition: all 0.3s ease;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M18 2H15C13.6739 2 12.4021 2.52678 11.4645 3.46447C10.5268 4.40215 10 5.67392 10 7V10H7V14H10V22H14V14H17L18 10H14V7C14 6.73478 14.1054 6.48043 14.2929 6.29289C14.4804 6.10536 14.7348 6 15 6H18V2Z" stroke="#2c3e50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </a>
            <a href="#" style="color: #2c3e50; text-decoration: none; transition: all 0.3s ease;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M23 3C22.0424 3.67548 20.9821 4.19211 19.86 4.53C19.2577 3.83751 18.4573 3.34669 17.567 3.12393C16.6767 2.90116 15.7395 2.9572 14.8821 3.28445C14.0247 3.61171 13.2884 4.1944 12.773 4.95372C12.2575 5.71303 11.9877 6.61234 12 7.53V8.53C10.2426 8.57557 8.50127 8.18581 6.93101 7.39545C5.36074 6.60508 4.01032 5.43864 3 4C3 4 -1 13 8 17C5.94053 18.398 3.48716 19.0989 1 19C10 24 21 19 21 7.5C20.9991 7.22145 20.9723 6.94359 20.92 6.67C21.9406 5.66349 22.6608 4.39271 23 3V3Z" stroke="#2c3e50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </a>
            <a href="#" style="color: #2c3e50; text-decoration: none; transition: all 0.3s ease;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M16 8C17.5913 8 19.1174 8.63214 20.2426 9.75736C21.3679 10.8826 22 12.4087 22 14V21H18V14C18 13.4696 17.7893 12.9609 17.4142 12.5858C17.0391 12.2107 16.5304 12 16 12C15.4696 12 14.9609 12.2107 14.5858 12.5858C14.2107 12.9609 14 13.4696 14 14V21H10V14C10 12.4087 10.6321 10.8826 11.7574 9.75736C12.8826 8.63214 14.4087 8 16 8V8Z" stroke="#2c3e50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M6 9H2V21H6V9Z" stroke="#2c3e50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M4 6C5.10457 6 6 5.10457 6 4C6 2.89543 5.10457 2 4 2C2.89543 2 2 2.89543 2 4C2 5.10457 2.89543 6 4 6Z" stroke="#2c3e50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </a>
        </div>
        <p style="color: #7f8c8d; margin: 0; font-size: 0.9rem;">
            Invoice Fraud Detection System v1.0 • Powered by Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display alert notifications if monitoring is active
    if hasattr(st.session_state.alert_system, 'running') and st.session_state.alert_system.running:
        alerts = st.session_state.alert_system.get_alerts(5)  # Get latest 5 alerts
        if alerts:
            for alert in reversed(alerts):  # Show newest first
                severity_class = f"alert-{alert['severity'].lower()}"
                st.markdown(f"""
                <div class="card {severity_class} alert-notification" style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="margin: 0; font-size: 1.1rem;">{alert['type']}</h4>
                            <p style="margin: 0.5rem 0 0; font-size: 0.9rem;">{alert['message']}</p>
                        </div>
                        <div style="text-align: right;">
                            <small style="color: #7f8c8d;">{alert['timestamp'].strftime('%H:%M:%S')}</small>
                            <div style="font-weight: bold; color: {'#e74c3c' if alert['severity'] == 'High' else '#f39c12' if alert['severity'] == 'Medium' else '#f1c40f'}">
                                {alert['severity']}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()