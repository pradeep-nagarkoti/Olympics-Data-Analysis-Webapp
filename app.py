import streamlit as st
import pandas as pd
import preprocessor, helper
import plotly.express as px
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.figure_factory as ff


df = pd.read_csv('data/athlete_events.csv')
region_df = pd.read_csv('data/noc_regions.csv')

df = preprocessor.preprocess(df, region_df)

st.sidebar.title('Olympics Data Analysis')
st.sidebar.image("E:\Olympic.png")
user_menu= st.sidebar.radio(
    'select an option',
    ('Medal Tally','Overall Analysis','Country-Wise Analysis','Athlete Wise Analysis','Medal Prediction')
)

if user_menu == 'Medal Tally':
    st.sidebar.header('Medal Tally')
    years, country = helper.country_year_list(df)

    selected_year = st.sidebar.selectbox('Select Year', years)
    selected_country = st.sidebar.selectbox('Select Country', country)
    medal_tally = helper.fetch_medal_tally(df, selected_year, selected_country)

    if selected_year == 'Overall' and selected_country == 'Overall':
     st.title('📊Overall Tally')
    if selected_year != 'Overall' and selected_country == 'Overall':
     st.title('Medal Tally in ' + str(selected_year) + ' Olympics')
    if selected_year == 'Overall' and selected_country != 'Overall':
     st.title(selected_country + ' Overall Performance')
    if selected_year != 'Overall' and selected_country != 'Overall':
     st.title(selected_country + ' Performance in ' + str(selected_year) + ' Olympics')

    st.table(medal_tally)

if user_menu == 'Overall Analysis':
    editions = df['Year'].unique().shape[0] - 1
    cities = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events = df['Event'].unique().shape[0]
    athletes = df['Name'].unique().shape[0]
    nations = df['region'].unique().shape[0]

    st.title('🔝Top Statistics')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header('Editions')
        st.title(editions)
    with col2:
        st.header('Hosts')
        st.title(cities)
    with col3:
        st.header('Sports')
        st.title(sports)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header('Events')
        st.title(events)
    with col2:
        st.header('Nations')
        st.title(nations)
    with col3:
        st.header('Athletes')
        st.title(athletes)

    nations_over_time = helper.data_over_time(df, 'region')
    fig = px.line(nations_over_time, x="Edition", y="region")
    st.title('🌍Participating Nations over the years')
    st.plotly_chart(fig)

    events_over_time = helper.data_over_time(df, 'Event')
    fig = px.line(events_over_time, x="Edition", y="Event")
    st.title('🏃Events over the years')
    st.plotly_chart(fig)

    athlete_over_time = helper.data_over_time(df, 'Name')
    fig = px.line(athlete_over_time, x="Edition", y="Name")
    st.title('📅Athletes over the years')
    st.plotly_chart(fig)

    st.title('🕒No. of Events over time (Every Sport)')
    fig, ax = plt.subplots(figsize=(25, 25))
    x = df.drop_duplicates(['Year', 'Sport', 'Event'])
    ax = sns.heatmap(
        x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype('int'),
        annot=True)
    st.pyplot(fig)

    st.title("🏆Most successful Athletes")
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    selected_sport = st.selectbox('Select a Sport', sport_list)
    x = helper.most_successful(df, selected_sport)
    st.table(x)

if user_menu == 'Country-Wise Analysis':
    st.sidebar.title('Country-wise Analysis')

    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()

    selected_country = st.sidebar.selectbox('Select a Country',country_list)

    country_df = helper.yearwise_medal_tally(df,selected_country)
    fig = px.line(country_df, x="Year", y="Medal")
    st.title(selected_country + " Medal Tally over the years")
    st.plotly_chart(fig)

    st.title(selected_country + " excels in the following sports")
    pt = helper.country_event_heatmap(df,selected_country)
    if pt is None or pt.empty:
        st.write("No data available for selected country.")
    else:
        # Heatmap plot
        fig, ax = plt.subplots(figsize=(20, 20))
        ax = sns.heatmap(pt, annot=True)
        st.pyplot(fig)

    st.title("Top 10 athletes of " + selected_country)
    top10_df = helper.most_successful_countrywise(df,selected_country)
    st.table(top10_df)

if user_menu == 'Athlete Wise Analysis':
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=1000, height=600, xaxis_title="Age", yaxis_title="Winning Chance")
    st.title("⏳Distribution of Age for winning medals")
    st.plotly_chart(fig)  


    x = []
    name = []
    famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']
    for sport in famous_sports:
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
        name.append(sport)

    fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=1000, height=600, xaxis_title="Age", yaxis_title="Gold Medalist")
    st.title("🥇Distribution of Age w.r.t Sports(Gold Medalist)")
    st.plotly_chart(fig)

    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    st.title('🧬Height Vs Weight')
    selected_sport = st.selectbox('Select a Sport', sport_list)

    temp_df = helper.weight_v_height(df, selected_sport)
    fig,ax = plt.subplots()
    ax = sns.scatterplot(x=temp_df['Weight'], y=temp_df['Height'], hue=temp_df['Medal'], style=temp_df['Sex'], s=10)
    st.pyplot(fig)   

    st.title("📈Men Vs Women Participation Over the Years")
    final = helper.men_vs_women(df)
    fig = px.line(final, x="Year", y=["Male", "Female"])
    fig.update_layout(autosize=False, width=1000, height=600)
    st.plotly_chart(fig)

if user_menu == 'Medal Prediction':
    st.title("🏅 Olympic Medal Prediction by Athlete Name")
    athlete_input = st.text_input("Enter the athlete's name:")

    if athlete_input:
        # Step 1: Clean and group data
        athlete_year_df = df.dropna(subset=['Name', 'Sex', 'Age', 'Height', 'Weight', 'Team', 'Year', 'Sport', 'Event', 'region'])
        athlete_year_df['Won_Medal'] = df['Medal'].notna().astype(int)

        athlete_year_df = athlete_year_df.groupby(['Name', 'Year'], as_index=False).agg({
            'Sex': 'first',
            'Age': 'mean',
            'Height': 'mean',
            'Weight': 'mean',
            'Sport': 'first',
            'Event': 'first',
            'region': 'first',
            'Won_Medal': 'max'
        })

        # Step 2: Encode entire DataFrame first
        from sklearn.preprocessing import LabelEncoder
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np

        le_sex = LabelEncoder()
        le_sport = LabelEncoder()
        le_event = LabelEncoder()
        le_region = LabelEncoder()

        athlete_year_df['Sex_encoded'] = le_sex.fit_transform(athlete_year_df['Sex'])
        athlete_year_df['Sport_encoded'] = le_sport.fit_transform(athlete_year_df['Sport'])
        athlete_year_df['Event_encoded'] = le_event.fit_transform(athlete_year_df['Event'])
        athlete_year_df['region_encoded'] = le_region.fit_transform(athlete_year_df['region'])

        # Step 3: Train model
        features = ['Age', 'Height', 'Weight', 'Sex_encoded', 'Sport_encoded', 'Event_encoded', 'region_encoded', 'Year']
        X = athlete_year_df[features]
        y = athlete_year_df['Won_Medal']

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Step 4: Get input row
        matches = athlete_year_df[athlete_year_df['Name'].str.lower().str.contains(athlete_input.lower())]

        if matches.empty:
            st.warning("Athlete not found.")
        else:
            latest_record = matches.sort_values('Year', ascending=False).iloc[0]

            try:
                input_data = np.array([[latest_record['Age'],
                                        latest_record['Height'],
                                        latest_record['Weight'],
                                        latest_record['Sex_encoded'],
                                        latest_record['Sport_encoded'],
                                        latest_record['Event_encoded'],
                                        latest_record['region_encoded'],
                                        latest_record['Year']]])

                prediction = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0][1]

                st.subheader(f"Athlete: {latest_record['Name']} ({latest_record['Year']})")
                st.write(f"Sport: {latest_record['Sport']} | Event: {latest_record['Event']} | Country: {latest_record['region']}")
                st.write(f"Age: {latest_record['Age']}, Height: {latest_record['Height']}, Weight: {latest_record['Weight']}")

                if prediction == 1:
                    st.success(f"✅ Likely to WIN a medal ({prob:.2%} confidence)")
                else:
                    st.error(f"❌ Unlikely to win a medal ({prob:.2%} confidence)")

            except Exception as e:
                st.error("Error during prediction. Try another athlete.")
                st.exception(e)