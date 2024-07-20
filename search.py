
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
        b: bool = self.generate_bool(f"Does this website seem to be useful?\nTitle:{title}\nDescription:{description}", save = False, max_attempt = 1)
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

    def analyze_v3_1(self, result: googlesearch.SearchResult, messages: list[dict[str, str]] = [])-> list[str]:
        markdown: str = self.analyze_load_website(result.url)
        instruction: str = "Pickup useful information to provide accurate information to the user."
        messages.append({ "role": "system", "content": markdown })
        return self.generate_list(instruction, messages = messages, save = False, max_attempt = 3)

    def execute_v3(self, plan: list[str]) -> tuple[str, list[googlesearch.SearchResult]]:
        result: dict[googlesearch.SearchResult, list[str]] = None

        while True:
            ic(plan)
            for p in plan:
                cmd = p.split()
                if cmd[0] == "plan":
                    plan = self.plan_dynamic(result)
                    break
                if cmd[0] == "search":
                    result_table: dict[str, int] = {"fast": 3, "medium": 5, "slow": 7}
                    num_results = result_table["medium"]

                    if len(cmd) > 1:
                        num_results = result_table[cmd[1]]

                    result = self.search_v3(num_results)
                elif cmd[0] == "summarize":
                    return self.summarize_v3(result), list(result.keys())

    def generate(
        self,
        messages: list[dict[str, str]] = [],
        verify: Callable[[str], bool] = lambda x: True,
        max_attempt: int = -1,
        save = True,
        save_func: Callable[[list[dict[str, str]], str], list[dict[str, str]]] = lambda x, y: x + [{ "role": "assistant", "content": y }],
        prefix: str = "",
        suffix: str = "",
        **kwargs
    ) -> str | None:
        attempt: int = 0

        while True:
            if attempt == max_attempt:
                return None
            attempt += 1

            response: openai.ChatCompletion = self.client.chat.completions.create(model = "gpt-3.5-turbo", messages = self.messages + messages, **kwargs)
            content: str = prefix + response.choices[0].message.content + suffix
            ic(content)

            if verify(content):
                ic(messages)
                if save:
                    self.messages += save_func(messages, content)
                return content
    
    def generate_xml(
            self,
            examples: str | list[str],
            root_tag: str,
            children: dict[str, Any],
            instruction: str = None,
            verify: Callable[[ET.Element], bool] = lambda x: True,
            messages: list[dict[str, str]] = [],
            save_func: Callable[[ET.Element], str] = lambda y: y,
            **kwargs
        ) -> ET.Element | None:
        def __save_func(x: list[dict[str, str]], y: str) -> list[dict[str, str]]:
            return messages + ([] if instruction is None else [{ "role": "system", "content": instruction}]) + [{ "role": "assistant", "content": save_func(ET.fromstring(y))}]

        attempt: int = 0

        __instruction: str
        __verify: Callable[[str], bool]
        __instruction, __verify = self.xml_instruction_and_verify(examples, root_tag, children, verify = verify)

        __messages: list[dict[str, str]] = []
        if instruction is not None:
            __messages.append({ "role": "system", "content": instruction })
        __messages.append({ "role": "system", "content": __instruction })

        content: str = self.generate(messages + __messages, __verify, suffix = f"</{root_tag}>", stop = f"</{root_tag}>", save_func = __save_func, **kwargs)
        ic("generate_xml", content)
        if content is None:
            return None
        return ET.fromstring(content)

    def generate_list(self, instruction: str, **kwargs) -> list[str] | None:
        def __save_func(y: ET.Element):
            return '\n'.join([f"* {yi.text}" for yi in y.findall("item")]).lstrip()

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

        element: ET.Element = self.generate_xml(example, "list", children, instruction, save_func = __save_func, **kwargs)
        if element is None:
            return None
        
        return [e.text for e in element.findall("item")]

    def generate_bool(self, instruction: str, max_attempt = -1, **kwargs) -> bool | None:
        def __verify(xml: ET.Element) -> bool:
            return xml.text.lower() in ["true", "false"]

        def __save_func(y: ET.Element) -> str:
            return y.text.lower()

        examples: list[str] = [ "<bool>True</bool>", "<bool>False</bool>" ]
        children: dict[str, str] = {}
        
        element: ET.Element = self.generate_xml(examples, "bool", children, instruction, verify = __verify, save_func = __save_func, max_attempt = 1, **kwargs)
        if element is None:
            return None

        text: str = element.text.lower()
        if text == "true":
            return True
        elif text == "false":
            return False

    def generate_string(self, instruction: str, **kwargs) -> str | None:
        example: str = "<string>String Value</string>"
        children: dict[str, str] = {}

        element: ET.Element = self.generate_xml(example, "string", children, instruction, save_func = lambda y: y.text, **kwargs)
        if element is None:
            return None

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

    def plan_v3(self, prompt: str) -> list[str]:
        messages: list[dict[str, str]] = [
            { "role": "user", "content": prompt },
            { "role": "system", "content": "Read the user input. Then answer the questions." }
        ]

        if self.generate_bool("Do you have unclear term or words in the user prompt?", messages = messages):
            return ["search fast", "plan"]

        if self.generate_bool("Do you need user interaction first to provide more accurate information?"):
            return ["plan"]

        if self.generate_bool("Can you provide accurate information about this?"):
            return ["summarize"]

        if self.generate_bool("Do you need user interaction after/during search online?"):
            return ["search", "plan"]

        return ["search", "summarize"]

    def plan_dynamic(self, past_result: dict[googlesearch.SearchResult, list[str]]) -> list[str]:
        temp_summary: str

        if past_result is None:
            temp_summary = "You have to clarify things the user wants to know first."
        else:
            past_result_str: str = ""
            for v in past_result.values():
                for vi in v:
                    past_result_str += f"{vi}\n"
            temp_summary: str = self.generate_string(f"{past_result_str}Read the keypoints above and write summary.", save = False)

        messages = [{ "role": "system", "content": temp_summary }]

        instruction: str = "Make a question for the user to provide more accurate or specific answer."

        question: str = self.generate_string(instruction)
        answer: str = input(f"{question}:")

        return self.plan_v3(answer)

    def run_v2(self) -> int:
        while True:
            prompt: int = input("Search > ")
            if prompt == "exit":
                break
            plan: list[str] = self.plan_v3(prompt)
            result: str
            websites: list[googlesearch.SearchResult]
            result, websites = self.execute_v3(plan)
            print(websites)
            print(result)
            ic(self.messages)

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
            result: Generator[googlesearch.SearchResult] = googlesearch.search(keyword= 5, advanced = True, sleep_interval = 1, timeout = 60)
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

    def search_v3(self, num_results: int) -> dict[googlesearch.SearchResult, list[str]]:
        keypoints: dict[searchresult.SearchResult, list[str]] = {}
        keypoints_str: str = ""
        keywords: list[str] = self.generate_list("Make search keywords to gather information online.")
        info: list[str] = self.generate_list("List stuff that you should collect from the internet.")

        for keyword in keywords:
            for result in googlesearch.search(keyword, num_results = num_results, advanced = True, timeout = 60):
                instruction: str = f"Title:{result.title}\nDescription:{result.title}\nDo you think this website contains necessary information?"
                if not self.generate_bool(instruction, save = False):
                    continue

                messages = [{ "role": "system", "content": keypoints_str }]

                result_keypoints: list[str] = self.analyze_v3_1(result, messages = messages)
                if result_keypoints is None:
                    continue
                keypoints[result] = result_keypoints
                
                keypoints_str += '\n'.join(result_keypoints).lstrip()
                messages[0]["content"] = keypoints_str

                if self.generate_bool("Is the information enough to explain the user's question?", save = False):
                    return keypoints

                if self.generate_bool(f"Is the information about keyword:{keyword} enough?", save = False):
                    break

        return keypoints

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
        messages: list[dict[str, str]] = []
        instruction: str = "Write the explanation from gathered information"

        if analyze_result is None:
            instruction = "Write the explanation without uncertain things."
        else:
            analyze_result_str: str = ""
            for k, p in analyze_result.items():
                for pi in p:
                    analyze_result_str += f"{pi}\n"
            messages.append({ "role": "system", "content": analyze_result_str })
            instruction = "Write the explanation from gathered information."

        return self.generate_string(instruction, messages = messages)

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
            return __verify_sub(xml, children) == __num_tags(xml) and verify(xml) if verify is not None else True
        return __verify

    def xml_example(self, examples: str | list[str], root_tag: str, children: dict[str, str | dict[str, Any]]) -> str:
        output: str = "You have to use XML format to write your response. Examples:\n"
        for i, ex in enumerate(examples if type(examples) == list else [examples]):
            output += f"Example {i}\n{ex}\n\n"

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

    def xml_instruction_and_verify(self, examples: str | list[str], root_tag: str, children: dict[str, Any], verify: Callable[[ET.Element], bool] = lambda x: True) -> (str, Callable[[str], bool]):
        instruction: str = self.xml_example(examples, root_tag, children)
        verify: Callable[[str], bool] = self.verify_xml_v2(root_tag, children, verify)
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

