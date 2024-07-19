
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

    def analyze_is_useful_v2(self, title: str, description: str) -> bool:
        b: bool = self.generate_bool(f"Does this website seem to be useful?\nTitle:{title}\nDescription:{description}", max_attempt = 1)
        return False if b is None else b

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

    def analyze_v3(self, search_result: list[googlesearch.SearchResult]) -> dict[googlesearch.SearchResult, list[str]]:
        instruction: str = "List the keypoints and the examples which needs to explain what the user want to know."
        points: dict[googlesearch.SearchResult, list[str]] = {}

        for result in search_result:
            if not self.analyze_is_useful_v2(result.title, result.description):
                continue

            website_md: str = self.analyze_load_website(result.url)
            if website_md is None:
                continue

            markdown_instruction: str = f"Read the Markdown below:\n{website_md}\n[End Markdown]\n{instruction}"
            points[result] = self.generate_list(markdown_instruction, max_attempt = 3)

        return points

    def execute_v2(self, plan: ET.Element) -> str | None:
        search_result: list[googlesearch.SearchResult] = []
        analyze_result: list[str] = []

        for t in plan.findall("action"):
            name: str = t.get("name")
            goal: str = t.get("goal")

            if name == "search":
                search_result = self.search_v2()
            elif name == "analyze":
                analyze_result = self.analyze_v3(search_result)
            elif name == "summarize":
                return self.summarize_v3(analyze_result)
        return None

    def generate(self, messages: list[dict[str, str]] = [], verify: Callable[[str], bool] = lambda x: True, max_attempt: int = -1, save = False, prefix: str = "", suffix: str = "", **kwargs) -> str | None:
        attempt: int = 0

        while True:
            if attempt == max_attempt:
                return None
            attempt += 1

            response: openai.ChatCompletion = self.client.chat.completions.create(model = "gpt-3.5-turbo", messages = self.messages + messages, **kwargs)
            content: str = prefix + response.choices[0].message.content + suffix
            ic(content)

            if verify(content):
                if save:
                    self.messages = messages + [{ "assistant", content }]
                return content
    
    def generate_xml(self, examples: str | list[str], root_tag: str, children: dict[str, Any], instruction: str = None, **kwargs) -> ET.Element | None:
        attempt: int = 0

        __instruction: str
        __verify: Callable[[str], bool]
        __instruction, __verify = self.xml_instruction_and_verify(examples, root_tag, children)

        messages: list[dict[str, str]] = []
        if instruction is not None:
            messages.append({ "role": "system", "content": instruction })
        messages.append({ "role": "system", "content": __instruction })

        content: str = self.generate(messages, __verify, suffix = f"</{root_tag}>", stop = f"</{root_tag}>", **kwargs)
        ic("generate_xml", content)
        element: ET.Element = ET.fromstring(content)

        return element

    def generate_list(self, instruction: str, **kwargs) -> list[str] | None:
        example: str = """
        <list>
            <item>Item 1</item>
            <item>Item 2</item>
            <item>Item 3</item>
        </list>
        """.strip()

        children: dict[str, str] = {
            "<item>": "item in the list"
        }

        __instruction: str = instruction + "\nYou have to make List in XML."

        element: ET.Element = self.generate_xml(example, "list", children, __instruction, **kwargs)
        
        return [e.text for e in element.findall("item")]

    def generate_bool(self, instruction: str, **kwargs) -> bool | None:
        examples: list[str] = [ "<value>True</value>", "<value>False</value>" ]
        children: dict[str, str] = {}
        __instruction: str = instruction + "\nYou have to make Bool value in XML"
        
        element: ET.Element = self.generate_xml(examples, "value", children, __instruction, **kwargs)
        text: str = element.text.lower()
        if text == "true":
            return True
        elif text == "false":
            return False
        return None

    def generate_string(self, instruction: str, **kwargs) -> str | None:
        example: str = "<value>String</value>"
        children: dict[str, str] = {}
        __instruction: str = instruction + "\nYou have to make String value in XML"

        element: ET.Element = self.generate_xml(example, "value", children, __instruction, **kwargs)
        text: str = element.text
        return text

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

    def search_v2(self) -> list[googlesearch.SearchResult]:
        instruction: str = "Extract search keywords for google search from previous interaction."
        keywords: list[str] = self.generate_list(instruction)
        ic(keywords)

        search_result: list[googlesearch.SearchResult] = []
        for keyword in keywords:
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

    def summarize_v3(self, analyze_result: dict[googlesearch.SearchResult, list[str]]) -> str:
        instruction: str = ""
        for k, p in analyze_result.items():
            for pi in p:
                instruction += f"{pi}\n"
        instruction += "Read the input above and summarize it."
        return self.generate_string(instruction)

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

