'''
Extract job and company information from indeed.com
Some code was adapted from: https://github.com/Joshmantova/Data_science_job_market_analysis_project/blob/master/src/indeed_web_scrape_script.py
'''

import requests
import time
import math
from bs4 import BeautifulSoup
import time
import pandas as pd

def extract_company_rating_from_company_page(soup):
    '''
    find the company rating score from the company page on Indeed
    '''
    company_rating = soup.find('span', attrs = {'class': 'css-htn3vt e1wnkr790'})
    if company_rating != None:
        company_rating = company_rating.text
    else:
        company_rating = soup.find('span', attrs = {'class': 'css-1n6k8zn e1wnkr790'})
        if company_rating != None:
            company_rating = company_rating.text
        

    print('Company Rating: {}'.format(company_rating))
    return company_rating

def extract_company_reviews_from_company_page(soup):
    '''
    find the company's number of reviews from the company page on Indeed
    '''
    company_number_of_reviews = soup.find('a', attrs = {'data-tn-element': 'reviews-countLink'})
    if company_number_of_reviews != None:
        company_number_of_reviews = company_number_of_reviews.text.split(' ')[-2]

    print('Number of reviews of the company: {}'.format(company_number_of_reviews))
    return company_number_of_reviews

def extract_company_reviews_from_job_page(soup):
    '''
    find the company's number of reviews from the job page on Indeed
    '''
    company_number_of_reviews = soup.find('div', attrs = {'class': 'icl-Ratings-count'})
    if company_number_of_reviews != None:
        company_number_of_reviews = company_number_of_reviews.text.split(' ')[-2]

    print('Number of reviews of the company: {}'.format(company_number_of_reviews))
    return company_number_of_reviews

def extract_company_year_from_company_page(soup):
    '''
    find the company's founded year from the company page on Indeed
    '''
    company_year = soup.find('li', attrs = {'data-testid': 'companyInfo-founded'})
    if company_year != None:
        company_year = company_year.find('div', attrs = {'class': 'css-1w0iwyp e1wnkr790'}).text

    print('The company was founded in: {}'.format(company_year))
    return company_year

def extract_company_size_from_company_page(soup):
    '''
    find the company size from the company page on Indeed
    '''
    company_size = soup.find('li', attrs = {'data-testid': 'companyInfo-employee'})
    if company_size != None:
        company_size = company_size.find('div', attrs = {'class': 'css-1w0iwyp e1wnkr790'}).text

    print('Company size: {}'.format(company_size))
    return company_size

def extract_company_revenue_from_company_page(soup):
    '''
    find the company revenue from the company page on Indeed
    '''
    company_revenue = soup.find('li', attrs = {'data-testid': 'companyInfo-revenue'})
    if company_revenue != None:
        company_revenue = company_revenue.find('div', attrs = {'class': 'css-1w0iwyp e1wnkr790'}).text

    print('Company revenue: {}'.format(company_revenue))
    return company_revenue

def extract_industry_from_company_page(soup):
    '''
    find which industry the company belongs to from the company page on Indeed
    '''
    industry = soup.find('li', attrs = {'data-testid': 'companyInfo-industry'})
    if industry != None:
        industry = industry.find('div', attrs = {'class': 'css-1w0iwyp e1wnkr790'}).text

    print('Industry: {}'.format(industry))
    return industry

def extract_job_title_from_job_page(soup):
    '''
    find the job title for each job posting from Indeed
    '''

    job_title =soup.find('h1', attrs = {'class': 'icl-u-xs-mb--xs icl-u-xs-mt--none jobsearch-JobInfoHeader-title'})
    if job_title != None:
        job_title = job_title.text
    else:
        job_title =soup.find('h1', attrs = {'class': 'iCIMS_Header'})
        if job_title != None:
            job_title = job_title.text
            
    print('Job Title: {}'.format(job_title))
    return job_title

def extract_company_from_job_page(soup):
    '''
    find the company name for each job posting from Indeed
    '''

    company = soup.find('div', attrs = {'class': 'icl-u-lg-mr--sm icl-u-xs-mr--xs'})

    if company != None:
        company = company.text
    print('Company: {}'.format(company))
    return company

def extract_location_from_job_page(soup):
    '''
    find the company location for each job posting from Indeed
    '''
    location = soup.find('div', attrs = {'class': 'icl-u-xs-mt--xs icl-u-textColor--secondary jobsearch-JobInfoHeader-subtitle jobsearch-DesktopStickyContainer-subtitle'})     

    if location != None:
        location = location.find_all('div')[-1]
    
    if location != None:
        location = location.text
    print('Location: {}'.format(location))
    return location

def extract_salary_from_job_page(soup):
    '''
    find the salary range for each job posting from Indeed
    '''

    salary = soup.find('span', attrs = {'class': 'icl-u-xs-mr--xs'})
    if salary != None:
        salary = salary.text

    print('Salary: {}'.format(salary))
    return salary

def extract_contract_type_from_job_page(soup):
    '''
    find the contract type for each job posting from Indeed
    '''
    #not able to locate the contract type information directly from page
    #first find salary info and then extract the next sibling (i.e., contract type) info
    salary = soup.find('span', attrs = {'class': 'icl-u-xs-mr--xs'})

    contract_type = None
    if salary != None:
        contract_type = salary.next_sibling
        if contract_type != None:
            contract_type = contract_type.text[4:]

    print('Contract type: {}'.format(contract_type))
    return contract_type

def extract_description_from_job_page(soup):
    '''
    find the descriptionfor each job posting from Indeed
    '''
    text_block = soup.find('div', attrs = {'class': 'jobsearch-jobDescriptionText'})

    text_elements = [t for t in text_block.find_all(text=True) if t.parent.name in ['p', 'b', 'li', 'div', 'i']]
    description = '\n '.join(text_elements)
    #print(description)
    return(description)
 
def parse_page(url):
    '''
    parse all information from an Indeed search researchpage
    '''
    url_list = [url]

    job_urls = []
    job_titles = []
    companies = []
    locations = []
    salaries = []
    contract_types = []
    descriptions = []

    company_urls = []
    company_ratings = []
    company_number_of_reviews = []
    company_founded_years = []
    company_sizes = []
    company_revenues = []
    industries = []

    i = 0
    for url in url_list:
        page = requests.get(url)
        soup = BeautifulSoup(page.text, features = 'html.parser')        

        for job_card in soup.find_all("a", id = lambda value: value and value.startswith("job_")):

            print('---------------------------')
            print(i)

            #obtain the url for each job and parse the webpage
           # job_url = f'https://uk.indeed.com' + job_card['href']
            job_url = f'https://www.indeed.com' + job_card['href']
            print(job_url)
            job_page = requests.get(job_url)
            job_soup = BeautifulSoup(job_page.text, features = 'html.parser')

            if extract_job_title_from_job_page(job_soup) != None:

                job_urls.append(job_url)
                job_titles.append(extract_job_title_from_job_page(job_soup))
                companies.append(extract_company_from_job_page(job_soup))

                locations.append(extract_location_from_job_page(job_soup))
                salaries.append(extract_salary_from_job_page(job_soup))
                contract_types.append(extract_contract_type_from_job_page(job_soup))
                descriptions.append(extract_description_from_job_page(job_soup))

                #obtain the url of the company page
                #company_url =  job_soup.find('a', href = lambda value: value and value.startswith("https://uk.indeed.com/cmp"))
                company_url =  job_soup.find('a', href = lambda value: value and value.startswith("https://www.indeed.com/cmp"))
            
                if company_url != None:
                    company_url = company_url['href']
                    company_urls.append(company_url)
                    company_page = requests.get(company_url)
                    company_soup = BeautifulSoup(company_page.text, features = 'html.parser')
        
                    company_ratings.append(extract_company_rating_from_company_page(company_soup))
                    #company_number_of_reviews.append(extract_company_reviews_from_company_page(company_soup))
                    company_number_of_reviews.append(extract_company_reviews_from_job_page(job_soup))
                
                    company_founded_years.append(extract_company_year_from_company_page(company_soup))
                    company_sizes.append(extract_company_size_from_company_page(company_soup))
                    company_revenues.append(extract_company_revenue_from_company_page(company_soup))
                    industries.append(extract_industry_from_company_page(company_soup))


                else:
                    company_urls.append(None)
                    company_ratings.append(None)
                    company_number_of_reviews.append(None)
                    company_founded_years.append(None)
                    company_sizes.append(None)
                    company_revenues.append(None)
                    industries.append(None)

                i += 1

        job_info = [job_titles, companies, locations, salaries, contract_types, descriptions, job_urls]
        company_info = [company_ratings, company_number_of_reviews, company_founded_years, company_sizes, company_revenues, industries, company_urls]
    
    return job_info, company_info


if __name__ == '__main__':
    search_area = 'CA'
    #start number is the starting number of the post in the search result. For example, 50 means the script will scrape from job 51-100 (on page 2)
    start_number = 50
    page_number = int(start_number/50)
    url = f'https://www.indeed.com/jobs?q=data+science&l=California&sort=date&limit=50&radius=25&start={str(start_number)}'

    job_info, company_info = parse_page(url)
    
    df = pd.DataFrame()
    df['Job_title'] = job_info[0]
    df['Company'] = job_info[1]
    df['Company_Rating'] = company_info[0]
    df['Number_of_Reviews_of_the_Company'] = company_info[1]
    df['Company_Founded_Year'] = company_info[2]
    df['Company_Size'] = company_info[3]
    df['Company_Revenue'] = company_info[4]
    df['Industry'] = company_info[5]
    df['Location'] = job_info[2]
    df['Salary'] = job_info[3]
    df['Contract_Type'] = job_info[4]
    df['Description'] = job_info[5]
    df['Job_URL'] = job_info[6]
    df['Company_URL'] = company_info[6]

    #df
    search_date = time.strftime("%Y%m%d")
    df.to_csv(f'df_{search_area}_{search_date}_{str(page_number)}.csv')
        



