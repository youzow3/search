
import argparse
import bs4
import googlesearch
from icecream import ic
import markdownify
import openai
import re
import requests_html
from typing import Any, Callable, Generator
import xml.etree.ElementTree as ET

class Application:
    def __init__(self):
        self.client: openai.OpenAI = openai.OpenAI(
            base_url = "http://localhost:8080/v1",
            api_key = "sk-no-key-required"
        )
        self.session: requests_html.HTMLSession = requests_html.HTMLSession()

        self.messages: list[dict[str, str]] = []

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

        tags: dict[str, str] = {
            "<useful>": "True or False, whether the website is useful or not.",
            "<reason>": "Describe why you write <useful> True or False."
        }

        xml_instruction: str
        verify: Callable[[str], bool]
        xml_instruction, verify = self.xml_instruction_and_verify(example, "website", tags)
        ic(xml_instruction)

        messages: list[dict[str, str]] = [
            { "role": "system", "content": f"Title:{title}\nDescription:{description}" },
            { "role": "system", "content": f"Read the website title and description. Then, decide the website is useful or not and write it.\n{xml_instruction}" }
        ]

        xml_str: str = self.generate(messages = messages, verify = verify, max_attempt = 2)
        if xml_str is None:
            return False
        
        xml: ET.Element = ET.fromstring(xml_str)
        return xml.find("useful").text == "True"

    def analyze_load_website(self, url: str) -> str:
        try:
            response: requests_html.HTMLResponse = self.session.get(url, timeout = 60)
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

    def analyze_v2(self, goal: str, search_result: list[googlesearch.SearchResult]) -> dict[googlesearch.SearchResult, tuple[list[str], list[str]]]:
        analyze_result: dict[googlesearch.SearchResult, tuple[list[str], list[str]]] = {}

        example: str = """
        Example:
        <points>
            <point>A is important module for C</point>
            <point>B can be used instead of A</point>

            <example>To replace A to B, run this command: command</example>
        </points>
        """.strip()

        tags: dict[str, str] = {
            "<point>": "Write a key point which is valuable for the user.",
            "<example>": "Write an example that describe a key point if the content provide it."
        }

        xml_instruction: str
        verify: Callable[[str], bool]
        xml_instruction, verify = self.xml_instruction_and_verify(example, "points", tags)
        ic(xml_instruction)

        for result in search_result:
            result_points: list[str] = []
            result_examples: list[str] = []

            if not self.analyze_is_useful(result.title, result.description):
                continue

            website_md: str = self.analyze_load_website(result.url)
            if website_md is None:
                continue

            messages: list[dict[str, str]] = [
                { "role": "system", "content": f"{website_md}" },
                { "role": "system", "content": f"Read the part of the website contents. Then, extract key points which are relevant to the user's question. In this analyze phase, you have to achieve the goal:\n{goal}\n{xml_instruction}" }
            ]

            xml_str: str = self.generate(messages = messages, verify = verify, max_attempt = 3)
            if xml_str is None:
                continue

            xml: ET.Element = ET.fromstring(xml_str)
            points: ET.Element = xml.findall("point")
            for p in points if points is not None else []:
                result_points.append(p.text)

            examples: ET.Element = xml.findall("example")
            for e in examples if examples is not None else []:
                result_examples.append(e.text)

            analyze_result[result] = (result_points, result_examples)

        return analyze_result

    def execute_v2(self, plan: ET.Element) -> str | None:
        search_result: list[googlesearch.SearchResult] = []
        analyze_result: list[str] = []

        for t in plan.findall("action"):
            name: str = t.get("name")
            goal: str = t.get("goal")

            if name == "search":
                search_result = self.search(goal)
            elif name == "analyze":
                analyze_result = self.analyze_v2(goal, search_result)
            elif name == "summarize":
                return self.summarize_v2(goal, analyze_result)
        return None

    def generate(self, messages: list[dict[str, str]] = [], verify: Callable[[str], bool] = lambda x: True, max_attempt: int = -1) -> str | None:
        attempt: int = 0

        while True:
            if attempt == max_attempt:
                return None
            attempt += 1

            response: openai.ChatCompletion = self.client.chat.completions.create(model = "gpt-3.5-turbo", messages = self.messages + messages)
            content: str = response.choices[0].message.content
            ic(content)

            if verify(content):
                return content
    
    def generate_xml(self, root_tag: str, children: dict[str, Any], instruction: str = None, max_attempt: int = -1) -> str | None:
        attempt: int = 0

        __instruction: str
        __verify: Callable[[str], bool]
        __instruction, __verify = xml_instruction_and_verify(root_tag, children)

        messages: list[dict[str, str]] = []
        if instruction is not None:
            messages.append({ "role": "system", "content": instruction })
        messages.append({ "role": "system", "content": __instruction })

        while True:
            if attempt == max_attempt:
                return None
            attempt += 1

            response: openai.ChatCompletion = self.client.chat.coompletions.create(model = "gpt-3.5-turbo", messages = self.messages + messages, stop = f"<{root_tag}>")
            content: str = response.choices[0].message.content
            ic(content)

            if __verify(content):
                return content

    def generate_xml_list(self, instruction: str, existing_list: str, max_attempt = -1) -> str | None:
        children: dict[str, str] = {
            "<item>": "item in the list"
        }

        __instruction: str = instruction + f"\nYou need to add items to the following list:\nf{existing_list}"

        return self.generate_xml("list", children, instruction, max_attempt = max_attempt)

    def plan_v2(self) -> ET.Element:
        example: str = """
        <plan>
            <action name="search" goal="get information about it" />
            <action name="analyze" goal="extract important stuff from search result" />
            <action name="summarize" goal="provide accurate information for the user" />
        </plan>
        """.strip()

        tags: dict[str, dict[str, str]] = {
            "<action>": {
                "name": "\"search\", \"analyze\" or \"summarize\".",
                "goal": "Brief description of the goal of this action." 
            }
        }

        xml_instruction: str
        verify: Callable[[str], bool]
        xml_instruction, verify = self.xml_instruction_and_verify(example, "plan", tags)
        ic(xml_instruction)

        self.messages += [
            { "role": "system", "content": f"From your understanding, plan how to get information from the internet.\n{xml_instruction}" }
        ]

        xml_str: str = self.generate(verify = verify)
        self.messages.append({ "role": "assistant", "content": xml_str })
        return ET.fromstring(xml_str)

    def run_v2(self) -> int:
        while True:
            prompt: int = input("Search > ")
            if prompt == "exit":
                break
            u: str = self.understand_v2(prompt)
            plan: ET.Element = self.plan_v2()
            result: str = self.execute_v2(plan)
            print(result)

    def search(self, goal: str) -> list[googlesearch.SearchResult]:
        example: str = """
        Example:
        <keywords>
            <keyword>foo</keyword>
            <keyword>bar</keyword>
        </keywords>
        """.strip()

        tags: dict[str, str] = {
            "<keyword>": "A search keyword for google search."
        }
        
        xml_instruction: str
        verify: Callable[[str], bool]
        xml_instruction, verify = self.xml_instruction_and_verify(example, "keywords", tags)
        ic(xml_instruction)

        self.messages += [
            { "role": "system", "content": f"Extract search keywords for google search from previous interaction. In this search phase, you have to achieve the goal:\n{goal}\n{xml_instruction}" }
        ]

        xml_str: str = self.generate(verify = verify)

        xml: ET.Element = ET.fromstring(xml_str)
        keyword_list: list[str] = [t.text for t in xml.findall("keyword")]

        search_result: list[googlesearch.SearchResult] = []
        for keyword in keyword_list:
            result: Generator[googlesearch.SearchResult] = googlesearch.search(keyword, num_results = 5, advanced = True, sleep_interval = 1, timeout = 60)
            _ = [search_result.append(r) for r in result]

        return search_result

    def summarize_v2(self, goal: str, analyze_result: dict[googlesearch.SearchResult, tuple[list[str], list[str]]]) -> str:
        example: str = """
        <summarize>
            <ref id="1" title="Website title 1" />
            <ref id="2" title="Website title 2" />
            <content>Summarized text</content>
        </summarize>
        """.strip()

        tags: dict[str, dict[str, str] | str] = {
            # "ref": "The website title which you use to make summary.",
            "<ref>": {
                "id": "The number to identify website which is quoted in content tag.",
                "title": "The website title"
            },
            "<content>": "Summary of search results. You should write quote like this:[1], [2], ... [n]"
        }
        
        xml_instruction: str
        xml_verify: Callable[[str], bool]
        xml_instruction, xml_verify = self.xml_instruction_and_verify(example, "summarize", tags)
        data: str = self.summarize_v2_website_data(analyze_result)
        self.messages += [
            { "role": "system", "content": data },
            { "role": "system", "content": f"Read the points and examples above. Points and examples are in <content> tag in each <website> tag. Then summarize them as final result. In this summarize phase, you have to achieve the goal:\n{goal}\n{xml_instruction}" }
        ]

        xml: ET.Element = ET.fromstring(self.generate(verify = xml_verify))
        content: ET.Element = xml.find("content")
        return content.text

    def summarize_v2_website_data(self, analyze_result: dict[googlesearch.SearchResult, tuple[list[str], list[str]]]) -> str:
        xml: ET.Element = ET.Element("websites") # Root element

        for result_key, result_data in analyze_result.items():
            result_xml: ET.Element = ET.Element("website")

            result_xml_title: ET.Element = ET.Element("title")
            result_xml_title.text = result_key.title

            result_xml_contents: ET.Element = ET.Element("contents")

            for result_data_point in result_data[0]:
                result_xml_point: ET.Element = ET.Element("point")
                result_xml_point.text = result_data_point
                result_xml_contents.append(result_xml_point)

            for result_data_example in result_data[1]:
                result_xml_example: ET.Element = ET.Element("example")
                result_xml_example.text = result_data_example
                result_xml_contents.append(result_xml_example)

            result_xml.append(result_xml_title)
            result_xml.append(result_xml_contents)

            xml.append(result_xml)

        return ET.tostring(xml)

    def understand_v2(self, prompt: str) -> None:
        example: str = """
        <understanding>
            <fact>the user wants to know how to do foo</fact>
            <inference>the user seems to want to know how the foo works.</inference>
        </understanding>
        """.strip()

        tags: dict[str, str] = {
            "<fact>": "The stuff the user wants to know which you can read from the given sentence.",
            "<inference>": "The stuff the user seems to want to know. This means the stuff you can infer from the given sentence."
        }

        xml_instruction: str
        verify: Callable[[str], bool]
        xml_instruction, verify = self.xml_instruction_and_verify(example, "understanding", tags)
        ic(xml_instruction)

        self.messages += [
            { "role": "user", "content": prompt },
            { "role": "system", "content": f"Read the user prompt. Then understand the user's intent.\n{xml_instruction}" }
        ]

        xml_str: str = self.generate(verify = verify)

        self.messages += [
            { "role": "assistant", "content": xml_str }
        ]

    def verify_xml(self, root_tag: str, verify: Callable[[ET.Element], bool] = lambda x: True) -> Callable[[str], bool]:
        def __convert(x: str) -> bool:
            try:
                xml: ET.Element = ET.fromstring(x)
                return verify(xml) if xml.tag == root_tag else False
            except:
                return False
        return __convert

    def verify_xml_v2(self, root_tag: str, children: dict[str, Any], verify: Callable[[ET.Element], bool] = lambda x: True) -> Callable[[str], bool]:
        def __verify(xml_str: str) -> bool:
            try:
                xml: ET.Element = ET.fromstring(xml_str)
                if xml.tag != root_tag:
                    return False
            except:
                return False

            def __num_tags(r: ET.Element) -> int:
                return sum([__num_tags(_r) for _r in r]) + len(r)

            def __verify_sub(sub: ET.Element, tags: dict[str, Any]) -> int:
                n_tags: int = 0
                for k, v in tags.items():
                    ic(k, v)
                    if not k.startswith('<') and not k.endswith('>'): # If it is not a tag
                        continue

                    k_items: list[ET.Element] = sub.findall(k.strip("<>"))
                    ic(k_items)
                    if k_items is None:
                        continue

                    n_tags += len(k_items)
                    
                    if type(v) == dict:
                        for k_item in k_items:
                            n_tags += __verify_sub(k_item, v)
                return n_tags

            ic(__verify_sub(xml, children), __num_tags(xml))
            return __verify_sub(xml, children) == __num_tags(xml)
        return __verify

    def xml_example(self, examples: str | list[str], root_tag: str, children: dict[str, str | dict[str, Any]]) -> str:
        output: str = "You have to use XML format to write your response. Examples:\n"
        for ex in examples if type(examples) == list else [examples]:
            output += f"{ex}\n"

        output += "Valid tags:\n"
        
        def __expand(__children: dict[str, Any], __output: str = "", __depth: int = 0) -> str:
            for k, v in __children.items():
                __output += '\t' * __depth + f"{k}:"

                if type(v) == str:
                    __output += f" {v}\n"
                elif type(v) == dict:
                    __output += '\n'
                    __output = __expand(v, __output, __depth + 1)
                else:
                    raise Exception()
            return __output

        return __expand(children, output)

    def xml_instruction_and_verify(self, examples: str | list[str], root_tag: str, children: dict[str, Any]) -> (str, Callable[[str], bool]):
        instruction: str = self.xml_example(examples, root_tag, children)
        verify: Callable[[str], bool] = self.verify_xml_v2(root_tag, children)
        return instruction, verify

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", default = 2, type = int)
    args: argparse.Namespace = parser.parse_args()

    app: Application = Application()
    exit_code: int
    if args.version == 2:
        exit_code = app.run_v2()
    else:
        print("Invalid version")
        exit_code = 1
    exit(exit_code)

