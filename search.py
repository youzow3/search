
import bs4
import googlesearch
from icecream import ic
import markdownify
import openai
import requests_html
from typing import Generator
import xml.etree.ElementTree as ET

class Application:
    def __init__(self):
        self.client: openai.OpenAI = openai.OpenAI(
            base_url = "http://localhost:8080/v1",
            api_key = "sk-no-key-required"
        )
        self.session: requests_html.HTMLSession = requests_html.HTMLSession()

        self.messages: list[dict[str, str]] = []

    def analyze(self, goal: str, search_result: list[googlesearch.SearchResult]) -> list[str]:
        important_points: list[str] = []

        for result in search_result:
            example: str = """
            Example 1:
            <website>
                <useful>True</useful>
                <reason>It seems to explain important stuff</reason>
            </website>
            Example 2:
            <website>
                <useful>False</useful>
                <reason>It doesn't seem to explain necessary stuff</reason>
            </website>
            """.strip()

            tags: str = """
            Valid tags in <website> are:
            <useful>: True or False, declare the website is useful or not.
            <reason>: String, describe why you write <useful> True, or False.
            """.strip()

            messages: list[dict[str, str]] = [
                { "role": "system", "content": f"Title:{result.title}\nDescription:{result.description}" },
                { "role": "system", "content": f"Read the website title and description. Then, decide the website is useful or not and write it. You have to use XML format to write result:\n{example}\n{tags}" }
            ]

            while True:
                response: openai.resources.chat.Completions = self.client.chat.completions.create(model = "phi-3-mini", messages = self.messages + messages)
                ic(response.choices[0].message.content)
                try:
                    xml: ET.Element = ET.fromstring(response.choices[0].message.content.replace('&', "&amp;"))
                    if xml.find("useful") is None:
                        continue
                    break
                except:
                    continue

            if xml.find("useful").text == "False":
                continue

            example = """
            Example:
            <points>
                <point>foo</point>
                <point>bar</point>

                <example>hoge hoge</example>
            </points>
            """.strip()

            tags = """
            Valid tags in <points>:
            <point>: String, write a key point in short sentences. The key point should be relevant to what the user want to know.
            <example>: String, write an example if the content provide it. \"example\" means some code in programming, for instance. Two or more <example> tags are not allowed.
            """.strip()

            website_md: str = self.analyze_load_website(result.url)
            if website_md is None:
                continue

            messages: list[dict[str, str]] = [
                { "role": "system", "content": f"{website_md}" },
                { "role": "system", "content": f"Read the part of the website contents. Then, extract key points which are relevant to the user's question. You have to use XML format to list key points:\n{example}\n{tags}" }
            ]

            for _ in range(10):
                response: openai.resources.chat.Completions = self.client.chat.completions.create(model = "phi-3-mini", messages = self.messages + messages)
                ic(response.choices[0].message.content)
                try:
                    xml: ET.Element = ET.fromstring(response.choices[0].message.content.replace('&', "&amp;"))
                    break
                except:
                    continue

            points = xml.findall("point")
            for p in points if points is not None else []:
                important_points.append(p.text)

            example_tag = xml.find("example")
            if example_tag is not None:
                important_points.append(example_tag.text)

        ic(important_points)
        return important_points

    def analyze_load_website(self, url: str) -> str:
        try:
            response: requests_html.HTMLResponse = self.session.get(url)
        except:
            return None

        if response.status_code != 200:
            return None

        soup: bs4.BeautifulSoup = bs4.BeautifulSoup(response.text, "html.parser")
        _ = [t.unwrap() for t in soup.find_all('a')]
        _ = [t.unwrap() for t in soup.find_all("img")]
        main_tag: bs4.element.Tag = soup.find("main")
        if main_tag == None:
            return None

        main_content: str = f"<html><body>{main_tag}</body></html>"
        return markdownify.markdownify(main_content)

    def execute(self, plan: ET.Element) -> str:
        search_result: list[googlesearch.SearchResult] = []
        analyze_result: list[str] = []

        for t in plan.findall("action"):
            name: str = t.get("name")
            goal: str = t.get("goal")

            if name == "search":
                search_result = self.search(goal)
            elif name == "analyze":
                analyze_result = self.analyze(goal, search_result)
            elif name == "summarize":
                return self.summarize(goal, analyze_result)

    def plan(self, understanding: str) -> ET.Element:
        example: str = """
        Example 1:
        <plan>
            <action name="search" goal="get information about it" />
            <action name="analyze" goal="extract important stuff from search result" />
            <action name="summarize" goal="provide accurate information for the user" />
        </plan>
        Example 2:
        <plan>
            <action name="search" goal="get basic information" />
            <action name="analyze" goal="extract basic knowledge" />
            <action name="search" goal="gather deeper explained information" />
            <action name="summarize" goal="provide easy-to-understand information" />
        </plan>
        """.strip()

        tags: str = """
        Valid tags in <plan>:
        <action>: Actions that system perform. Variables are:
            name: \"search\", \"analyze\" or \"summarize\" are valid.
                * analyze should be placed after search action.
                * summarize should be placed after analyze.
                * search can be used any position.
            goal: The goal for this action.
        """.strip()

        self.messages += [
            { "role": "assistant", "content": understanding },
            { "role": "system", "content": f"From your understanding, plan how to get information from the internet.\nYou have to use XML format to make plan:\n{example}\n{tags}\n" }
        ]
 
        while True:
            response: openai.resources.chat.Completions = self.client.chat.completions.create(
                model = "phi-3-mini",
                messages = self.messages
            )
            ic(response.choices[0].message.content)

            try:
                xml: ET.Element = ET.fromstring(response.choices[0].message.content.replace('&', "&amp;"))
                xml_valid_tag: list[ET.Element] = list(filter(lambda x: x.get("name") in ["search", "analyze", "summarize"], xml.findall("action")))
                if len(xml_valid_tag) == 0:
                    continue
                break
            except:
                continue

        return xml

    def run(self) -> int:
        while True:
            prompt = input("Search > ")
            u: str = self.understand(prompt)
            plan: ET.Element = self.plan(u)
            result = self.execute(plan)
            print(result)
        return 0

    def search(self, goal: str) -> list[googlesearch.SearchResult]:
        example: str = """
        Example:
        <search>
            <keywords>
                <keyword>foo</keyword>
                <keyword>bar</keyword>
            </keywords>
        </search>
        """.strip()

        tags: str = """
        Valid tags in <keywords> in <search>:
        <keyword>: String, the search keyword for google search.
        """
        self.messages += [
            { "role": "system", "content": f"Extract search keywords for google search from previous interaction. In this search phase, you have to achieve the goal:\n{goal}\nYou have to use XML format to make search keywords list:\n{example}\n{tags}" }
        ]

        while True:
            response: openai.resources.chat.Completions = self.client.chat.completions.create(model = "phi-3-mini", messages = self.messages)
            ic(response.choices[0].message.content)
            try:
                keywords: ET.Element = ET.fromstring(response.choices[0].message.content.replace('&', "&amp;"))
                break
            except:
                continue

        keywords_tag: ET.Element = keywords.find("keywords")
        keyword_list: list[str] = [t.text for t in keywords_tag.findall("keyword")]

        search_result: list[googlesearch.SearchResult] = []
        for keyword in keyword_list:
            result: Generator[googlesearch.SearchResult] = googlesearch.search(keyword, advanced = True, sleep_interval = 1, timeout = 60)
            _ = [search_result.append(r) for r in result]

        return search_result

    def summarize(self, goal:str, analyze_result: list[str]) -> str:
        keypoints: str = '\n'.join(analyze_result)
        messages: list[dict[str, str]] = [
            { "role": "system", "content": keypoints },
            { "role": "system", "content": "Read the key points above. Then summarize them as final result."}
        ]

        response: openai.resources.chat.ChatCompletions = self.client.chat.completions.create(model = "phi-3-mini", messages = self.messages + messages)
        return response.choices[0].message.content

    def understand(self, prompt: str) -> str:
        self.messages += [
            { "role": "user", "content": prompt },
            { "role": "system", "content": "Read the user prompt. Then understand the user's intent, and write it. Remember, assumptions cause misinformation for the user, so before writing, think carefully." }
        ]

        response: openai.resources.chat.Completions = self.client.chat.completions.create(
            model = "phi-3-mini",
            messages = self.messages
        )

        return response.choices[0].message.content

if __name__ == "__main__":
    exit(Application().run())
