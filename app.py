'''
Combine multiple csv files in a folder into one
'''

import streamlit as st
import pandas as pd
import numpy as np
import docx2txt
import pdfplumber
import spacy_streamlit as ss
import spacy
import re
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from flashtext import KeywordProcessor
from wordcloud import WordCloud
from PIL import Image

#nlp = spacy.load('en_core_web_sm')
st.set_page_config(layout = 'wide')
colour_scale = ['#ebfff7', '#b4e8d7', '#a3dac8', '#92ccba', '#81beab', '#71b19d', '#60a38f', '#4f9682', '#3d8974', '#2a7c67']

def read_data_from_csv():
    # Will use the integrated dataset when it is ready
    #filename = './Indeed_Elroy_2018/indeed_job_dataset.csv'
    filename = './consolidated_integrated_df_20211021.csv'
    df = pd.read_csv(filename, index_col = None, header = 0)
    #df = df[df['Skill'].notna()]

    df['Country'] = 'United States'
    df.loc[(df.Region == 'California'), 'Country'] = 'United States'
    df.loc[(df.Region == 'UK'), 'Country'] = 'United Kingdom'

    df.loc[(df.Inferred_Job_Type == 'data_analyst'), 'Inferred_Job_Type'] = 'Data Analyst'
    df.loc[(df.Inferred_Job_Type == 'data_engineer'), 'Inferred_Job_Type'] = 'Data Engineer'
    df.loc[(df.Inferred_Job_Type == 'data_scientist'), 'Inferred_Job_Type'] = 'Data Scientist'
    
    df = df[df['Consolidated_Skills'].notna()]
    #df.drop(columns = ['python', 'sql', 'machine learning', 'r', 'hadoop', 'tableau', 'sas', 'spark', 'java','Others'], inplace = True)
    #df.drop(columns = ['Consulting and Business Services','Internet and Software', 'Banks and Financial Services', 'Health Care', 'Insurance', 'Other_industries'], inplace = True)
    #df.drop(columns = ['CA', 'NY', 'VA', 'TX', 'MA', 'IL', 'WA', 'MD', 'DC', 'NC', 'Other_states'], inplace = True)
    return df

def get_skill_counts(skills, industry, job_type, JD_skill_lists, JD_industry_list, JD_job_type_list):
    '''
    get a list of skills and their counts
    '''

    counts_per_skill = [0] * len(skills)
    
    for i in range(len(JD_skill_lists)):
        if industry == 'All' or JD_industry_list[i] == industry:
            if job_type == 'All' or JD_job_type_list[i] == job_type:
                for j in range(len(JD_skill_lists[i])):
                    index = skills.index(JD_skill_lists[i][j])
                    counts_per_skill[index] += 1
                    
    return counts_per_skill

def get_skill_lists(df):
    '''
    get the counts of skills based on industry and job type
    '''
    
    #JD_skill_lists = df["Skill"].tolist()
    JD_skill_lists = df["Consolidated_Skills"].tolist()
    

    for i in range(len(JD_skill_lists)):
        JD_skill_lists[i] = JD_skill_lists[i][1: -1]
        JD_skill_lists[i] = JD_skill_lists[i].replace("'", "")
        JD_skill_lists[i] = JD_skill_lists[i].split(',')
        for j in range(len(JD_skill_lists[i])):
            JD_skill_lists[i][j] = JD_skill_lists[i][j].strip()

    # extract the list of skills
    skills = []
    
    for i in range(len(JD_skill_lists)):
        for j in range(len(JD_skill_lists[i])):
            if JD_skill_lists[i][j] not in skills:
                skills.append(JD_skill_lists[i][j])

   # JD_industry_list = df["Industry"].tolist()
    JD_industry_list = df["Adjusted_Industry"].tolist()
    #JD_job_type_list = df["Job_Type"].tolist()
    JD_job_type_list = df["Inferred_Job_Type"].tolist()

    #unique_industry = sorted(df["Industry"].dropna().unique().tolist())
    #unique_job_type = sorted(df["Job_Type"].dropna().unique().tolist())

    unique_industry = sorted(df["Adjusted_Industry"].dropna().unique().tolist())
    unique_job_type = sorted(df["Inferred_Job_Type"].dropna().unique().tolist())

    unique_industry.insert(0, 'All')
    unique_job_type.insert(0, 'All')
    
    #get the counts of skills
    #industry = 'all'
    #job_type = 'all'
    df_skill_counts = pd.DataFrame()
    df_skill_counts['Consolidated_Skills'] = skills
    
    for industry in unique_industry:
        for job_type in unique_job_type:
            counts_per_skill = get_skill_counts(skills, industry, job_type, JD_skill_lists, JD_industry_list, JD_job_type_list)
            column_name = industry + '+' + job_type
            df_skill_counts[column_name] = counts_per_skill
    
    return skills, unique_industry, unique_job_type, df_skill_counts

def extract_skills_from_file(raw_text, all_skills):
    keywordprocessor = KeywordProcessor()
    skill_to_search = all_skills
    
   # for i in range(len(all_skills)):
   #     skill = all_skills[i]
        # seperate the words that contain mixed cases
    #    if not skill.isupper() and not skill.islower(): 
   #         skill_with_space = re.sub(r"(?<=\w)([A-Z])", r" \1", skill)
   #         skill_to_search.append(skill_with_space)

    for i in range(len(skill_to_search)):
        keywordprocessor.add_keyword(skill_to_search[i])

    skills_extracted = keywordprocessor.extract_keywords(raw_text)
    
    return list(set(skills_extracted))

def update_labels(labels):
    updated_labels = labels
    for i in range(len(updated_labels)):
        if updated_labels[i] == 'NaturalLanguageProcessing':
            updated_labels[i] = 'NLP'
    
    return updated_labels

def compute_skill_occurrence(df, skills, industry_option, job_type_option):
    #JD_skill_lists = df["Skill"].tolist()

    JD_skill_lists = df["Consolidated_Skills"].tolist()

    for i in range(len(JD_skill_lists)):
        JD_skill_lists[i] = JD_skill_lists[i][1: -1]
        JD_skill_lists[i] = JD_skill_lists[i].replace("'", "")
        JD_skill_lists[i] = JD_skill_lists[i].split(',')
        for j in range(len(JD_skill_lists[i])):
            JD_skill_lists[i][j] = JD_skill_lists[i][j].strip()

   # JD_industry_list = df["Industry"].tolist()
    JD_industry_list = df["Adjusted_Industry"].tolist()
    #JD_job_type_list = df["Job_Type"].tolist()
    JD_job_type_list = df["Inferred_Job_Type"].tolist()
    
    occurrence_matrix = [ [ 0 for i in range(len(skills)) ] for j in range(len(skills)) ]

    for i in range(len(skills)):
        for j in range(i +1, len(skills)):
            skill_1 = skills[i]
            skill_2 = skills[j]

            for m in range(len(JD_skill_lists)):
                if industry_option == 'All' or JD_industry_list[m] == industry_option:
                    if job_type_option == 'All' or JD_job_type_list[m] == job_type_option:
                        if skill_1 in JD_skill_lists[m] and skill_2 in JD_skill_lists[m]:
                            occurrence_matrix[i][j] += 1
                            occurrence_matrix[j][i] += 1

    return occurrence_matrix
    
def construct_dashboard():
    st.subheader('Understand the data science job market')
    instruction_text = '<p style="color:#71b19d; font-size: 17px;">All charts shown are interactive! Place your mouse cursor over the charts to see the plotted values and labels.</p>'
    st.markdown(instruction_text, unsafe_allow_html=True)
    df = read_data_from_csv()
    
    #df.rename(columns={'Company_Industry':'Industry'}, inplace=True)
    skills, unique_industry, unique_job_type, df_skill_counts = get_skill_lists(df)

    # add dropdown menus for selecting industry and job type options
    col3_0, col3_1, col3_2 = st.columns(3)
    columns_3 = [col3_0, col3_1, col3_2]
    
    country_option = col3_0.selectbox("Country", ['All', 'United Kingdom', 'United States'])
    industry_option = col3_1.selectbox("Industry", unique_industry)
    job_type_option = col3_2.selectbox("Job Type", unique_job_type)
    options = [country_option, industry_option, job_type_option]
    combined_option = industry_option + '+' + job_type_option

    # add donut charts for displaying the percentages
    groups = ['Country', 'Adjusted_Industry', 'Inferred_Job_Type']
    groups_to_show = ['Country', 'Industry', 'Job_Type']
    
    for i in range(len(groups)):
        df_group = df.groupby([groups[i]]).size().sort_values(ascending = False).reset_index()
        df_group = df_group.set_axis(['Group', 'Count'], axis='columns', inplace = False)

        labels = df_group['Group'].tolist()
        values = df_group['Count'].tolist()

        number_of_slides = [2, len(unique_industry), len(unique_job_type)]
        marker_colour = ['#95cdba',] * number_of_slides[i]

        # highlight the selected category
        if options[i] != 'All':       
            index = labels.index(options[i])
            #marker_colour[index] = '#3d8974'
            marker_colour[index] = '#F8C53A'
                       
        # Use `hole` to create a donut-like pie chart
        fig_donut_industry = go.Figure(data=[go.Pie(labels = labels, values = values, hole=.5)])
        fig_donut_industry.update_traces(textposition = 'inside', marker = dict(colors = marker_colour,
                                                                          line=dict(color='#EBF5F1', width=1)))
        fig_donut_industry.update_layout(uniformtext_minsize =12.5, uniformtext_mode = 'hide',
                                     showlegend = False, margin=dict(l = 50, r = 50, t = 50, b = 50),
                                     annotations=[dict(text = groups_to_show[i].replace('_', ' '), font_size=20, showarrow=False)])

        columns_3[i].plotly_chart(fig_donut_industry, use_container_width = True)

    text_to_write = 'Our database contains __' + str(len(df)) + '__ data science job postings strapped from www.indeed.com. '
    text_to_write += 'The job postings were published by companies from __' + str(len(unique_industry) - 1) + '__ different industries in the United States and United Kingdom. '
    text_to_write += 'There are __' + str(len(unique_job_type) - 1) + '__ types of jobs in the database. '
    text_to_write += 'The required skills for each job were extracted from the job description using a deep learning model. '
    text_to_write += 'In total there are __' + str(len(skills)) + '__ data science skills were extracted from our job posting data. '

    st.markdown(text_to_write)
    col2_0, col2_1 = st.columns(2)

    if country_option != 'All':
        filter_df = df.loc[df['Country'] == country_option]
        skills, unique_industry, unique_job_type, df_skill_counts = get_skill_lists(filter_df)

    df_to_display = df_skill_counts.sort_values(combined_option, ascending = False)

    colors = ['#95cdba',] * 20

    #skill_labels = df_to_display['Skill'].tolist()[:20]
    skill_labels = df_to_display['Consolidated_Skills'].tolist()[:20]

    skill_labels = update_labels(skill_labels)

    fig_ranking = go.Figure(data=[go.Bar(x = df_to_display[combined_option].tolist()[:20] ,
                                         y = skill_labels, orientation='h', marker_color = colors)])

    #title_text = 'The Top 20 Skills for ' + job_type_option + ' in ' +industry_option'
    title_text = 'The Top 20 Data Science Skills Required by the Job Market.'
    
    if job_type_option == 'All' and industry_option != 'All':
        title_text = 'The Top 20 Data Science Skills in ' + industry_option + '.'

    if job_type_option != 'All' and industry_option == 'All':
        title_text = 'The Top 20 Data Science Skills for ' + job_type_option + '.'

    if job_type_option != 'All' and industry_option != 'All':
        title_text = 'The Top 20 Skills for ' + job_type_option + ' in ' + industry_option +  '.'

    title_text += ' <br><sub>Ranking of skills based on their frequency of appearance (horizontal axis) in job postings data collected from Indeed.com.</sub>'
    
    fig_ranking.update_layout(title = title_text, yaxis = dict(tickmode = 'linear', autorange="reversed"))

    fig_ranking.update_yaxes(tickfont = dict(size = 12))
    fig_ranking.update_xaxes(showgrid = True, gridwidth=1.5, gridcolor='White')
    
     #fig_ranking = go.Figure(data=[go.Bar(x = df_to_display[combined_option].tolist()[:20] , y = y_axis_labels, text = y_axis_labels, orientation='h', marker_color = colors)])
    #fig_ranking.update_yaxes(tickfont = dict(size = 12), visible=False)
    #fig_ranking.update_traces(textposition='outside')
    #fig_ranking.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

    fig_ranking.update_layout(height = 600)
    col2_0.plotly_chart(fig_ranking, use_container_width = True)

    #z  = np.random.rand(20, 20)
    #occurrence_matrix = compute_skill_occurrence(df, df_to_display['Skill'].tolist()[:20], industry_option, job_type_option)
    occurrence_matrix = compute_skill_occurrence(df, df_to_display['Consolidated_Skills'].tolist()[:20], industry_option, job_type_option)
    
    fig_heatmap = go.Figure(data = go.Heatmap(z = occurrence_matrix, x = skill_labels, y = skill_labels, colorscale = colour_scale))

    title_text = 'Frequency of Co-currence of The Top 20 Skills in Job Descriptions.'
    title_text += ' <br><sub>Heat map for the frequency of joint appearance of skills pairs. The colour shading reflects frequency of appearance.</sub>'

    fig_heatmap.update_layout(title = title_text, yaxis = dict(tickmode = 'linear', autorange="reversed"))
    fig_heatmap.update_layout(height = 650)
    col2_1.plotly_chart(fig_heatmap, use_container_width = True)
    

def generate_wordcloud(df_skill_counts, industry, job_type, options):
    #options = [x.replace(' ', '') for x in options]
    combined = industry + '+' + job_type
    df_for_counts = df_skill_counts.sort_values(combined, ascending = False)

    d = {}
    for index, row in df_for_counts.iterrows():
        #skill = row['Skill']
        skill = row['Consolidated_Skills']
        #skill = skill.replace(' ', '')
        count = int(row[combined])
        d[skill] = count

    def color_func(word, *args, **kwargs):
        if word in options:
            color = '#60A38F'
        else:
            color = '#abb8b3'
        return color

    wordcloud = WordCloud(background_color = 'white', colormap='summer', random_state = 1, width = 400, height = 300).generate_from_frequencies(frequencies=d)
    return(wordcloud)

def find_career_recommedations(options, df):
    #st.write(''.join(options))
    #options = [x.replace(' ', '') for x in options]
    options = [x.strip() for x in options]
    skills, unique_industry, unique_job_type, df_skill_counts = get_skill_lists(df)

    data = []
    for i in range(len(unique_industry)):
        industry = unique_industry[i]
        for j in range(len(unique_job_type)):
            job_type = unique_job_type[j]

            if industry != 'All' and job_type !=  'All':
                combined = industry + '+' + job_type
                df_for_counts = df_skill_counts.sort_values(combined, ascending = False)
                #top_20_skills = df_for_counts['Skill'].tolist()[:20]
                top_20_skills = df_for_counts['Consolidated_Skills'].tolist()[:20]
                top_20_skills = [x.strip() for x in top_20_skills]
            
                number_of_intersection = len(list(set(top_20_skills) & set(options)))
                score = float(number_of_intersection)/20

                data.append([industry, job_type, top_20_skills, score])

    df_scores = pd.DataFrame.from_records(data)
    df_scores.columns = ['Industry', 'Job Type', 'Core Skills', 'Score']
    df_scores = df_scores.sort_values('Score', ascending = False)
    df_scores = df_scores.reset_index(drop = True)

    instruction = 'Here are the job matches for you! Each Word Cloud illustrates the required skills for a specific job type and industry. '
    instruction += 'Font size reflects the importance of each skill.'
    st.write(instruction)

    # show overall matches to the job types

    if st.checkbox('Show overall matches to job types', value = True):

        col3_0, col3_1, col3_2 = st.columns(3)
        columns_3 = [col3_0, col3_1, col3_2]

        for i in range(3):
            columns_3[i].success('Overall Match: ' +  unique_job_type[i+1])
            combined = 'All+' + unique_job_type[i+1]
            df_for_counts = df_skill_counts.sort_values(combined, ascending = False)
            top_20_skills = df_for_counts['Consolidated_Skills'].tolist()[:20]
            top_20_skills = [x.strip() for x in top_20_skills]
            
            number_of_intersection = len(list(set(top_20_skills) & set(options)))
            score = float(number_of_intersection)/20
            columns_3[i].markdown('<p>    Match score: ' + str(round(score, 2)) +  '</p>', unsafe_allow_html = True)

            wordcloud = generate_wordcloud(df_skill_counts, 'All', unique_job_type[i+1], options)

            fig = plt.figure(1, figsize=(12, 12))
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.show()
            columns_3[i].pyplot(fig, use_container_width = True)
            
            skill_gap = list(set(top_20_skills) - set(options))
            skill_gap = ", ".join(skill_gap)
            columns_3[i].markdown('<p>    Potential skill gap: ' + skill_gap + '</p>', unsafe_allow_html = True)

    if st.checkbox('Show top matches to job types and industries', value = False):
        col3_0, col3_1, col3_2 = st.columns(3)
        columns_3 = [col3_0, col3_1, col3_2]

        for i in range(3):
            industry = df_scores['Industry'].iloc[i]
            job_type = df_scores['Job Type'].iloc[i]
            score = df_scores['Score'].iloc[i]

            skill_gap = list(set(df_scores['Core Skills'].iloc[i]) - set(options))
            skill_gap = ", ".join(skill_gap)
        
            columns_3[i].success('Best Match ' + str(i+1) + '/3')
            columns_3[i].markdown('<p>    Industry: ' + industry +  '<br>' + '    Job type: ' + job_type
                                  +  '<br>' + '    Match score: ' + str(round(score, 2)) +  '</p>', unsafe_allow_html = True)
            wordcloud = generate_wordcloud(df_skill_counts, industry, job_type, options)

            fig = plt.figure(1, figsize=(12, 12))
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.show()
            columns_3[i].pyplot(fig, use_container_width = True)

            columns_3[i].markdown('<p>    Potential skill gap: ' + skill_gap + '</p>', unsafe_allow_html = True)
        

def construct_career_finder():
    st.subheader('Match your skills to a career')

    df = read_data_from_csv()
    #df.rename(columns={'Company_Industry':'Industry'}, inplace=True)
    skills, unique_industry, unique_job_type, df_skill_counts = get_skill_lists(df)

    docx_file = st.file_uploader('Upload your CV or resume', type = ['pdf', 'docx'])
    
    skill_extracted = []

    if 'button_clicked' not in st.session_state:
        st.session_state['button_clicked'] = 0
    if 'recommendation_clicked' not in st.session_state:
        st.session_state['recommendation_clicked'] = 0

    if st.button('Extract skills from file'):
        st.session_state['button_clicked'] = 1
    
    if st.session_state['button_clicked']:
        if docx_file is not None:
            if docx_file.type == 'application/pdf':
                # process pdf file
                try:
                    with pdfplumber.open(docx_file) as pdf:
                        pages = pdf.pages[0]
                        raw_text = pages.extract_text()
                        
                except:
                    st.write('None')
            # process word file
            else:
                raw_text = docx2txt.process(docx_file)

            skills_to_extract = skills
            skill_extracted = extract_skills_from_file(raw_text, skills_to_extract)

    options = st.multiselect( 'Your skills extracted from the file. You can also manually add or remove skills in the box below:', skills, skill_extracted)

    if st.checkbox('Show file content', value = False):
        st.markdown(raw_text)
        
    if st.button('Find your career match'):
        st.session_state['recommendation_clicked'] = 1
        
    if st.session_state['recommendation_clicked']:    
        find_career_recommedations(options, df)

def contruct_about():
    st.subheader('About the project')
    description = 'There is a lack of available knowledge on how to best prepare for a job in the Data Science industry. '
    description += 'Also, there is no clear path for progressing in a data science career path within an industry or across different industries. '
    description += 'As with other fields, a clear path would help an applicant understand the skills and qualities classified as suitable for roles in the data science field, '
    description += 'with reference to any existing relevant experiences. '
    description += ' A clear path will also provide an applicant with an estimated salary that can be negotiated for a role of interest in the data science field. '
    st.write(description)
    description = 'We developed this tool to identify your skill gaps and make auto-suggestions for career transition and promotion in data science related jobs.'
    st.write(description)
    st.subheader('Who we are')
    st.write('We are project team 36 of the Data Science for All (DS4A) Women 2021 program! ')
    image = Image.open('./Photos/Team_Photo2.png')
    st.image(image, use_column_width = True)




def main():

    st.title('Find Your Data Science Career Path')
    menu = ['Job Market Dashboard', 'Career Path Finder', 'About']
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == 'Job Market Dashboard':
        construct_dashboard()
    elif choice == 'Career Path Finder':
        construct_career_finder()
    else:
        contruct_about()

if __name__ == '__main__':
    main()
