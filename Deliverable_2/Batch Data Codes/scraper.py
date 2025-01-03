"""Scraping module for scraping articles from websites."""
import json
import re

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from tqdm import tqdm
from datetime import datetime
import random

class NotSupportedWebsiteException(Exception):
    """
    Exception for not supported websites.
    It should be raised when the user passes an URL that is not supported by our app.
    """
    def __init__(self, message="The website from the provided URL is not supported."):
        super().__init__(message)


class Scraper:
    """
    This class encompasses all the scraping functionality.
    The main functionalities are done by:
    - scrape_article_urls: getting the urls of the articles from the main page 
                            or some topic page of a news outlet website    
    - scrape_content: scraping the title and content of an article from its url

    There are also some helper functions:
    - get_site_variables_dict: loading the config of the news outlet websites
    - discern_website_from_url: discerning the website from the article url
                                this is done by checking the url against the
                                patterns defined in the config
    """
    def __init__(self, path_to_site_variables: str):
        self.site_variables_dict = self.get_site_variables_dict(path_to_site_variables)

    @staticmethod
    def get_site_variables_dict(path: str) -> dict:
        """
        Read JSON file config and return it as a dictionary.
        The JSON file contains the config for the news outlet websites.
        """
        with open(path, "r", encoding="utf-8") as f:
            site_variables_dict = json.load(f)
        return site_variables_dict


    def scrape_article_urls(self, main_url: str) -> list[str]:
        """
        Get the list of article urls from the main page 
        or some topic page of a news outlet website.
        """
        try:
          response = requests.get(main_url, timeout=5)
        except:
          print(f"Failed for {main_url}")
          raise Exception(f"Failed in scrape articles urls for {main_url}")
        
        
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise Exception(  # pylint: disable=broad-exception-raised
                f"HTTP request failed with status code {response.status_code}"
            ) from e

        soup = BeautifulSoup(response.text, "html.parser")

        site_variables = self.discern_website_from_url(main_url)

        url_matchings = site_variables.get("url_matchings")
        compiled_excluded_patterns = []
        compiled_included_patterns = []
        # compile regex patterns
        if url_matchings.get("excluded_patterns"):
            compiled_excluded_patterns = [
                re.compile(pattern.replace("\\\\", "\\"))
                for pattern in url_matchings["excluded_patterns"]
            ]
        if url_matchings.get("included_patterns"):
            compiled_included_patterns = [
                re.compile(pattern.replace("\\\\", "\\"))
                for pattern in url_matchings["included_patterns"]
            ]

        all_hrefs = [a["href"] for a in soup.find_all("a", href=True)]
        url_matchings = site_variables["url_matchings"]
        matching_hrefs = [
            f"https://thesun.co.uk{href}" if href.startswith("/") else href
            for href in all_hrefs
            if self.check_href_match_condition(
                href,
                url_matchings,
                compiled_excluded_patterns,
                compiled_included_patterns,
            )
        ]
        matching_hrefs = list(dict.fromkeys(matching_hrefs))
        return matching_hrefs

    def discern_website_from_url(self, url: str) -> dict:
        """
        Discern website from an url.
        This is done by checking the url against the patterns defined in the config.
        If the url matches a pattern, the website is discerned.
        Otherwise, an exception is raised.
        """
        if url.startswith("https://www.cbssports.com"):
            return self.site_variables_dict["cbsnews"]

        site_variables = next(
            (
                site_dict
                for key, site_dict in self.site_variables_dict.items()
                if url.startswith(f"https://www.{key}")
                or url.startswith(f"https://{key}")
            ),
            None,
        )
        if not site_variables:
            raise NotSupportedWebsiteException()
        return site_variables

    @staticmethod
    def check_href_match_condition(
        url, url_matchings, compiled_excluded_patterns, compiled_included_patterns
    ):
        """Check href match condition."""
        excluded_patterns_match_condition = True
        if compiled_excluded_patterns:
            excluded_patterns_match_condition = not any(
                pattern.match(url) for pattern in compiled_excluded_patterns
            )

        included_patterns_match_condition = False
        if compiled_included_patterns:
            included_patterns_match_condition = any(
                pattern.match(url) for pattern in compiled_included_patterns
            )

        starts_with_match_condition = True
        if url_matchings.get("starts_with"):
            starts_with_match_condition = any(
                url.startswith(starts_with)
                for starts_with in url_matchings["starts_with"]
            )
        return (
            starts_with_match_condition and excluded_patterns_match_condition
        ) or included_patterns_match_condition

    @staticmethod
    def scrape_content(
        url: str,
        paragraph_tag: str,
        title_tag: str,
        date_tag: str = None,
        keywords_tag: str = None,
        potential_premium_tag: str = None,
        premium_str: str = None,
        exclude=None,
        include=None,
    ) -> dict[str, str]:
        """Scrape content."""
        if exclude is None:
            exclude = []

        try:
          response = requests.get(url, timeout=15)
          response.raise_for_status()
        except:
          raise Exception(f"Failed scraping for {url}")


        

        soup = BeautifulSoup(response.text, "html.parser")
        # If include is not None, then I want to include the first element that matches the include tags with its children
        if include:
            for tag in include:
                if soup.find(class_=tag):
                    soup = soup.find(class_=tag)
                    break
        
        # Find everything in a list what is in class 'keyword'
        if keywords_tag != "" and keywords_tag != None:
            keywords = soup.find_all(class_=keywords_tag)
            keywords = [keyword.text for keyword in keywords]
        else:
            keywords = []

        content = []
        paragraphs = soup.find_all(paragraph_tag)
        for paragraph in paragraphs:

            element_class = paragraph.get("class", None)
            common_class = {}
            if element_class and exclude:
                common_class = set(element_class).intersection(exclude)

            element_id = paragraph.get("id", None)
            common_id = {}
            if element_id and exclude:
                common_id = set(element_id).intersection(exclude)

            if (
                paragraph.find_parent(class_=exclude)
                or paragraph.find_parent(id=exclude)
                or common_class
                or common_id
            ):
                continue

            content.append(paragraph.text.strip())
        content = " ".join(content)

        special_chars_trans = str.maketrans(
            {"\n": " ", "\xa0": " ", "\t": " ", "'": "'"}
        )

        # Find em of class 'date' and take its text
        # If there is 'datetime' attribute then take its value
        date = None
        date = soup.find("time")
        if date:
            date = date.get("datetime")
        else:
            date = soup.find(class_=date_tag)
            if date:
                date = date.text.strip()
        # date = soup.find(class_=date_tag)
        # if date:
        #     date = date.text.strip()
        # Find all headers 
        headers = soup.find_all(["h1"])
        # Search if there is div with class 'wrap' and has strong text 'DALSZA CZĘŚĆ ARTYKUŁU JEST DOSTĘPNA DLA SUBSKRYBENTÓW' if yes then remove it
        is_premium = False
        if potential_premium_tag != None and potential_premium_tag != "" and premium_str != None and premium_str != "":
            potential_premiums = soup.find_all(class_=potential_premium_tag)
            for premium in potential_premiums:
                if premium:
                    if premium_str in premium.text:
                        # premium.decompose()
                        is_premium = True
                        break
        # Remove header from soup if it has a class or id that is in the exclude list
        for header in headers:
            element_class = header.get("class", None)
            common_class = {}
            if element_class and exclude:
                common_class = set(element_class).intersection(exclude)

            element_id = header.get("id", None)
            common_id = {}
            if element_id and exclude:
                common_id = set(element_id).intersection(exclude)

            if (
                header.find_parent(class_=exclude)
                or header.find_parent(id=exclude)
                or common_class
                or common_id
            ):
                header.decompose()
                
        if soup.find(title_tag) is None:
            title = soup.find(class_=title_tag)
            if title:
                title = title.text
            else:
                title = url.split("/")[-1].replace("-", " ").replace(".html", "")
        else:
            title = soup.find(title_tag).text

        return {
            "title": title.translate(special_chars_trans).strip() if title else None,
            "content": content.translate(special_chars_trans).strip()
            if content
            else None,
            "date": date if date else None,
            "keywords": keywords,
            "is_premium": is_premium,
        }

    def scrape(self, url: str):
        """Scrape content."""
        site_dict = self.discern_website_from_url(url)
        result_dict = self.scrape_content(
            url,
            site_dict["paragraph_tag"],
            site_dict["title_tag"],
            date_tag=site_dict["date_tag"],
            keywords_tag=site_dict["keywords_tag"],
            potential_premium_tag=site_dict["potential_premium_tag"],
            premium_str=site_dict["premium_str"],
            exclude=site_dict["exclude"],
            include=site_dict["include"],
        )
        result_dict["source_site"] = site_dict["source_site"]
        return result_dict

def scrape_all(site_variables_dict: dict, pages=10):
    '''
    Function to scrape all articles from the site_variables_dict. The articles are scraped and input into the database.
    '''

    # Assert pages are between 1 and 10
    assert 1 <= pages <= 10

    # Scrape urls
    urls = []
    for ws in site_variables_dict:
        for i, site in site_variables_dict[ws]['topics'].items():
            for j, ps in enumerate(site_variables_dict[ws]['page_suffix']):
                print(f"Scraping urls for topic: {i} and page: {j+1}")
                scraper = Scraper("variables_dict.json")
                try:
                  urls += scraper.scrape_article_urls(f"{site}{ps}/")
                except:
                  urls = urls
                print(f"Scraped {len(urls)} urls")
                if j+1 == pages:
                    break

    results = __get_results_from_urls(urls=urls, scraper=scraper)
    
    return results

def __get_results_from_urls(urls: list[str], scraper: Scraper):
    '''
    Function to get results from urls. The results are scraped from the urls and returned as a dictionary.
    '''
    # pool = connect_unix_socket()
    # unique_ids = read_unique_ids(hdfs_host=hdfs_host, folder_path=folder_path, file_name="unique_ids.txt")
    # unique_ids = list(unique_ids)   
    # with pool.connect() as conn:
    #     query = f"""
    #     SELECT url_text FROM {table_name};
    #     """
    #     result = conn.execute(text(query))
    #     # Get all unique urls from db as list
    #     db_urls = conn.execute(text(query))
    # # db_urls = c.fetchall()
    # # conn.close()
    # db_urls = [url[0] for url in db_urls]
    # Scrape content and use progress bar
    results = {}
    for url in tqdm(urls):
        # if url in results or url in unique_ids:
        #     print(f"Skipping {url} as it has already been scraped...")
        #     continue
        results[url] = scraper.scrape(url)
    
    for url, result in results.items():
        # Replace '\n' with ' ' in date
        if result['date']:
            result['date'] = result['date'].replace('\n', ' ')
        if 'Dodano:' in result['date']:
            result['date'] = result['date'].split('Dodano:')[1].strip()
        if result['source_site'] == 'wnp.pl':
            # Transform date to datetime from format '06-03-2024 05:57' to format 'yyyy-MM-ddTHH:mm:ss.fffZ'
            result['date'] = datetime.strptime(result['date'], '%d-%m-%Y %H:%M').strftime("%Y-%m-%d %H:%M:00")
        elif result['source_site'] == 'wysokienapiecie.pl':
            # Transform date to datetime from format '2023-05-08 06:47:31' to format 'yyyy-MM-ddTHH:mm:ss.fffZ'
            result['date'] = datetime.strptime(result['date'], '%Y-%m-%d %H:%M:%S').strftime("%Y-%m-%d %H:%M:00")
            
            result['keywords'] = [keyword[1:] for keyword in result['keywords']]
        elif result['source_site'] == 'beurs.nl':
            # Remove the first word (day name) from the input date string
            input_date = ' '.join(result['date'].split()[1:])

            # Define the mapping of Dutch month names to numerical representations
            month_mapping = {
                'januari': '01',
                'februari': '02',
                'maart': '03',
                'april': '04',
                'mei': '05',
                'juni': '06',
                'juli': '07',
                'augustus': '08',
                'september': '09',
                'oktober': '10',
                'november': '11',
                'december': '12'
            }

            # Replace Dutch month names with numerical representations
            for dutch_month, numeric_month in month_mapping.items():
                input_date = input_date.replace(dutch_month, numeric_month)
            
            input_format = "%d %m %Y %H:%M"
            parsed_date = datetime.strptime(input_date, input_format)
            output_format = "%Y-%m-%d %H:%M:%S"
            result['date'] = parsed_date.strftime(output_format)

    # set the keys of the dict to be the numbers and add previous keys as values 'url'
    results2 = {i: dict(results[url], **{'url': url}) for i, url in enumerate(results)}
    # Change the name of the key 'content' to 'text' and add random
    for i, result in results2.items():
        results2[i]['text'] = results2[i].pop('content')
        results2[i]['random'] = str(random.randint(0, 1000000))
    
    return results2