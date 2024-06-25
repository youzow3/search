
import bs4
import googlesearch
from icecream import ic
import markdownify
import openai
import requests_html
from typing import Callable, Generator
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

        example: str = """
        Example:
        <points>
            <point>foo</point>
            <point>bar</point>

            <example>hoge hoge</example>
        </points>
        """.strip()

        tags: str = """
        Valid tags in <points>:
        <point>: String, write a key point in short sentences. The key point should be relevant to what the user want to know.
        <example>: String, write an example if the content provide it. \"example\" means some code in programming, for instance. Two or more <example> tags are not allowed.
        """.strip()

        for result in search_result:
            if not self.analyze_is_useful(result.title, result.description):
                continue

            website_md: str = self.analyze_load_website(result.url)
            if website_md is None:
                continue

            messages: list[dict[str, str]] = [
                { "role": "system", "content": f"{website_md}" },
                { "role": "system", "content": f"Read the part of the website contents. Then, extract key points which are relevant to the user's question. In this analyze phase, you have to achieve the goal:\n{goal}\nYou have to use XML format to list key points:\n{example}\n{tags}\nNOTE: DO NOT use any tags other than those described above." }
            ]

            xml_str = self.generate(messages = messages, verify = self.verify_xml("points"), max_attempt = 3)
            if xml_str is None:
                continue

            xml = ET.fromstring(xml_str)
            points = xml.findall("point")
            for p in points if points is not None else []:
                important_points.append(p.text)

            example_tag = xml.find("example")
            if example_tag is not None:
                important_points.append(example_tag.text)

        ic(important_points)
        return important_points

    def analyze_is_useful(self, title: str, description: str) -> bool:
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
            { "role": "system", "content": f"Title:{title}\nDescription:{description}" },
            { "role": "system", "content": f"Read the website title and description. Then, decide the website is useful or not and write it. You have to use XML format to write result:\n{example}\n{tags}\nNOTE: DO NOT use any tags other than those described above." }
        ]

        def __verify(xml: ET.Element) -> bool:
            return (xml.find("useful") is not None) and (xml.find("reason") is not None)

        xml_str: str = self.generate(messages = messages, verify = self.verify_xml("website", __verify), max_attempt = 2)
        if xml_str is None:
            return False
        
        xml = ET.fromstring(xml_str)
        return xml.find("useful").text == "True"

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
        return None

    def generate(self, messages: list[dict[str, str]] = [], verify: Callable[[str], bool] = lambda x: True, max_attempt: int = -1) -> str:
        attempt = 0

        while True:
            if attempt == max_attempt:
                return None
            attempt += 1

            response: openai.ChatCompletion = self.client.chat.completions.create(model = "gpt-3.5-turbo", messages = self.messages + messages)
            content: str = response.choices[0].message.content
            ic(content)

            if verify(content):
                return content
            
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
            name: \"search\", \"analyze\" or \"summarize\"
                * analyze should be placed after search action.
                * summarize should be placed after analyze.
                * search can be used any position.
            goal: The goal for this action.
            NOTE: \"search\", \"analyze\" and \" summarize\" should be appeared at least 1 times each.
        """.strip()

        self.messages += [
            { "role": "system", "content": f"From your understanding, plan how to get information from the internet.\nYou have to use XML format to make plan:\n{example}\n{tags}\nNOTE: DO NOT use any other tags than those listed above." }
        ]
 
        def __verify(xml: ET.Element) -> bool:
            is_search: bool = False
            is_analyze: bool = False
            is_summarize: bool = False

            for action in xml.findall("action"):
                name = action.get("name")
                if name == "search":
                    is_search = True
                elif name == "analyze":
                    is_analyze = True
                elif name == "summarize":
                    is_summarize = True
            return is_search and is_analyze and is_summarize

        xml_str: str = self.generate(verify = self.verify_xml("plan", __verify))
        self.messages.append({ "role": "assistant", "content": xml_str })
        return ET.fromstring(xml_str)

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
        <keywords>
            <keyword>foo</keyword>
            <keyword>bar</keyword>
        </keywords>
        """.strip()

        tags: str = """
        Valid tags in <keywords>:
        <keyword>: String, the search keyword for google search.
        """
        self.messages += [
            { "role": "system", "content": f"Extract search keywords for google search from previous interaction. In this search phase, you have to achieve the goal:\n{goal}\nYou have to use XML format to make search keywords list:\n{example}\n{tags}\nNOTE: DO NOT use any other tags other than those described above." }
        ]

        def __verify(xml: ET.Element) -> bool:
            if len(xml.findall("keyword")) > 0:
                return True
            return False

        xml_str: str = self.generate(verify = self.verify_xml("keywords", __verify))

        xml: ET.Element = ET.fromstring(xml_str)
        keyword_list: list[str] = [t.text for t in xml.findall("keyword")]

        search_result: list[googlesearch.SearchResult] = []
        for keyword in keyword_list:
            result: Generator[googlesearch.SearchResult] = googlesearch.search(keyword, advanced = True, sleep_interval = 1, timeout = 60)
            _ = [search_result.append(r) for r in result]

        return search_result

    def summarize(self, goal: str, analyze_result: list[str]) -> str:
        keypoints: str = '\n'.join(analyze_result)
        self.messages += [
            { "role": "system", "content": keypoints },
            { "role": "system", "content": "Read the key points above. Then summarize them as final result. In this summarize phase, you have to achieve the goal:\n{goal}" }
        ]

        return self.generate()

    def understand(self, prompt: str) -> str:
        self.messages += [
            { "role": "user", "content": prompt },
            { "role": "system", "content": "Read the user prompt. Then understand the user's intent. Remember, assumptions cause misinformation for the user, so before writing, think carefully." }
        ]

        output: str = self.generate()
        self.messages.append({ "role": "assistant", "content": output })
        return output

    def understand_v2(self, prompt: str) -> (list[str], list[str]):
        example: str = """
        <understanding>
            <fact>the user wants to know how to do foo</fact>
            <inference>the user seems to want to know how the foo works.</inference>
        </understanding>
        """.strip()
        
        tags: str = """
        Valid tags in <understanding>:
        <fact>: String, The stuff the user wants to know which you can read from the given sentence.
        <inference>: String, The stuff the user seems to want to know. This means the stuff you can infer from the given sentence.
        """

        self.messages += [
            { "role": "user", "content": prompt },
            { "role": "system", "content": "Read the user prompt. Then understand the user's intent. You have to use XML format to write understanding:\n{example}\n{tags}.\nNOTE: DO NOT use any other tags other than those described above." }
        ]

        def __verify(xml: ET.Element) -> bool:
            facts: list[ET.Element] = xml.findall("fact")
            inference: list[ET.Element] = xml.findall("inference")
            return (len(facts if facts is not None else []) + len(inference if inference is not None else [])) > 0

        xml_str: str = self.generate(verify = self.verify_xml("understanding", __verify))
        xml: ET.Element = ET.fromstring(xml_str)

        facts: list[str] = [t.text for t in xml.findall("fact")]
        inferences: list[str] = [t.text for t in xml.findall("inference")]

        self.messages.append({ "role": "assistant", "content": xml_str })

        return facts, inferences

    def verify_xml(self, root_tag: str, verify: Callable[[ET.Element], bool] = lambda x: True) -> Callable[[str], bool]:
        def __convert(x: str) -> bool:
            try:
                xml: ET.Element = ET.fromstring(x)
                return verify(xml) if xml.tag == root_tag else False
            except:
                return False
        return __convert


if __name__ == "__main__":
    exit(Application().run())
