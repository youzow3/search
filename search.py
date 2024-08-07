
import argparse
import bs4
import googlesearch
import logging
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

        self.logger: logging.Logger = logging.getLogger(f"{__name__}.Application")

        formatter: logging.Formatter = logging.Formatter(fmt = "%(levelname)s: %(funcName)s: %(message)s")
        stream_handler: logging.StreamHandler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        file_handler: logging.FileHandler = logging.FileHandler("search.py.log")
        file_handler.setFormatter(formatter)

        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)

        self.session: requests_html.HTMLSession = requests_html.HTMLSession()

        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": "This is automated web research system. You have to follow the instructions system provide strictly."}
        ]
        
        self.qa_response: list[dict[str, str]] = []

    def analyze_is_useful(self, title: str, description: str) -> bool:
        b: bool = self.generate_bool(f"Does this website seem to be useful?\nTitle:{title}\nDescription:{description}", save = False, max_attempt = 1)
        return False if b is None else b

    def analyze_load_website(self, url: str) -> str:
        self.logger.debug("Start loading website")
        self.logger.debug(f"url = {url}")
        try:
            response: requests_html.HTMLResponse = self.session.get(url, timeout = 60)
        except Exception as e:
            self.logger.debug(str(e))
            self.logger.debug(f"Failed to get data from {url}")
            return None

        if response.status_code != 200:
            self.logger.debug(f"Failed to get data from {url}. Status code: {response.status_code}")
            return None

        soup: bs4.BeautifulSoup = bs4.BeautifulSoup(response.text, "html.parser")
        _ = [t.unwrap() for t in soup.find_all('a')]
        _ = [t.unwrap() for t in soup.find_all("img")]
        main_tag: bs4.element.Tag = soup.find("main")
        if main_tag == None:
            self.logger.debug(f"Failed to find <main> tag.")
            return None

        main_content: str = f"<html><body>{main_tag}</body></html>"
        markdown_content: str = markdownify.markdownify(main_content)
        self.logger.debug("Finished loading website")
        return markdown_content

    def analyze(self, result: googlesearch.SearchResult, messages: list[dict[str, str]] = [])-> list[str]:
        markdown: str = self.analyze_load_website(result.url)
        if markdown is None:
            return None

        instruction: str = "Pickup useful information to provide accurate information to the user."
        messages.append({ "role": "system", "content": markdown })
        return self.generate_list(instruction, messages = messages, save = False, max_attempt = 3)

    def execute(self, plan: list[str]) -> tuple[str, list[googlesearch.SearchResult]]:
        result: dict[googlesearch.SearchResult, list[str]] = None

        while True:
            for p in plan:
                cmd = p.split()
                if cmd[0] == "plan":
                    plan = self.plan_dynamic(result)
                    self.logger.info(f"Plan -> {plan}")
                    break
                if cmd[0] == "search":
                    result_table: dict[str, int] = {"fast": 3, "medium": 5, "slow": 7}
                    num_results = result_table["medium"]

                    if len(cmd) > 1:
                        num_results = result_table[cmd[1]]

                    _result: dict[googlesearch.SearchResult, list[str]] = self.search(num_results)
                    if result is None:
                        result = _result
                    else:
                        result |= _result

                elif cmd[0] == "summarize":
                    self.logger.debug(result)
                    return self.summarize(result), None if result is None else list(result.keys())

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
        self.logger.debug("Start generating AI response")
        self.logger.debug(f"messages = {messages}")
        self.logger.debug(f"max_attempt = {max_attempt}")
        self.logger.debug(f"save = {save}")
        self.logger.debug(f"prefix = {prefix}")
        self.logger.debug(f"suffix = {suffix}")

        while True:
            if attempt == max_attempt:
                self.logger.debug(f"Reached max_attempt ( = {max_attempt})")
                self.logger.debug("Failed to generate AI response")
                return None
            attempt += 1
            self.logger.debug(f"Attempt {attempt}")

            response: openai.ChatCompletion = self.client.chat.completions.create(model = "gpt-3.5-turbo", messages = self.messages + messages, **kwargs)
            content: str = prefix + response.choices[0].message.content + suffix
            self.logger.debug(f"Response:\n{content}")

            if verify(content):
                if save:
                    self.messages += save_func(messages, content)

                self.logger.debug("Finished generating AI response")
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
            return x[:-1] + [{ "role": "assistant", "content": save_func(ET.fromstring(y))}]

        attempt: int = 0

        __instruction: str
        __verify: Callable[[str], bool]
        __instruction, __verify = self.xml_instruction_and_verify(examples, root_tag, children, verify = verify)

        __messages: list[dict[str, str]] = []
        if instruction is not None:
            __messages.append({ "role": "system", "content": instruction })
        __messages.append({ "role": "system", "content": __instruction })

        content: str = self.generate(messages + __messages, __verify, suffix = f"</{root_tag}>", stop = f"</{root_tag}>", save_func = __save_func, **kwargs)
        if content is None:
            return None

        return ET.fromstring(content)

    def generate_list(self, instruction: str, **kwargs) -> list[str] | None:
        def __save_func(y: ET.Element):
            return '\n'.join([f"* {yi.text}" for yi in y.findall("item")]).lstrip()

        example: str = '\n'.join([
            "<list>",
            "\t<item>Item 1</item>",
            "\t<item>Item 2</item>",
            "\t<item>Item 3</item>",
            "</list>"
        ])

        children: dict[str, str] = {
            "<item>": "item in the list"
        }

        element: ET.Element = self.generate_xml(example, "list", children, instruction, save_func = __save_func, **kwargs)
        if element is None:
            return None
        
        l: list = [e.text for e in element.findall("item")]
        return None if l == [] else l

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
        examples: str = ["<string>\n\tString Value\n</string>", "<string>\n\tInstruction-based String Value\n</string>"]
        children: dict[str, str] = {}

        element: ET.Element = self.generate_xml(examples, "string", children, instruction, save_func = lambda y: y.text, **kwargs)
        if element is None:
            return None

        text: str = '\n'.join(list(map(lambda x: x.removeprefix("\t"), element.text.split('\n'))))
        return text

    def generate_string_freely(self, instruction: str, messages: list[dict[str, str]] = [], **kwargs) -> str | None:
        messages.append({ "role": "system", "content": instruction })

        return self.generate(messages = messages, **kwargs)

    def plan(self, prompt: str) -> list[str]:
        self.messages.append({ "role": "user", "content": prompt })
        

        if self.generate_bool("Do you have unclear term or words in the user prompt?"):
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
                past_result_str += "\n* ".join(v).lstrip()
            temp_summary: str = self.generate_string(f"{past_result_str}\nRead the keypoints above and write summary.", save = False)

        messages: list[dict[str, str]] = [{ "role": "system", "content": temp_summary }]

        instruction: str = "Make a question for the user to provide more accurate or specific answer."

        question: str = self.generate_string(instruction)
        answer: str = input(f"{question}:")

        return self.plan(answer)

    def run(self, args: argparse.Namespace) -> int:
        log_levels = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }

        if args.log_level.lower() in log_levels.keys():
            self.logger.setLevel(log_levels[args.log_level.lower()])
        else:
            self.logger.warning("Log level is invalid. Use \"info\" for this time.")
            self.logger.setLevel(logging.INFO)
            
        while True:
            prompt: int = input("Search > ")
            if prompt == "exit":
                break
            plan: list[str] = self.plan(prompt)
            result: str
            websites: list[googlesearch.SearchResult]
            result, websites = self.execute(plan)
            self.logger.debug(self.messages)
            if websites is None:
                print("No websites were used to provide this information.")
            else:
                print("\n\n".join(list(map(lambda x: f"{x.title}\nURL: {x.url}", websites))).strip())
            print()
            print(result)
            
            self.update_qa_response(prompt, result)

    def search(self, num_results: int) -> dict[googlesearch.SearchResult, list[str]]:
        def __n_keywords(num_results: int) -> int:
            if num_results < 5:
                return 1
            elif num_results < 7:
                return 3
            else:
                return 5

        keypoints: dict[searchresult.SearchResult, list[str]] = {}
        keypoints_str: str = None
        info: list[str] = self.generate_list("List stuff that you should collect from the internet.")
        n_keywords = __n_keywords(num_results)
        keywords: list[str] = self.generate_list("Make search keywords to gather information online. The max number of keywords is {n_keywords}")[:n_keywords]

        self.logger.info(f"Search keywords list -> {keywords}")

        for keyword in keywords:
            for result in googlesearch.search(keyword, num_results = num_results, advanced = True, timeout = 60):
                instruction: str = f"Title:{result.title}\nDescription:{result.title}\nDo you think this website contains necessary information?"
                if not self.generate_bool(instruction, save = False):
                    continue

                messages: list[dict[str, str]] = [] if keypoints_str is None else [{ "role": "system", "content": keypoints_str }]

                result_keypoints: list[str] = self.analyze(result, messages = messages)
                if result_keypoints is None:
                    continue
                keypoints[result] = result_keypoints

                if keypoints_str is None:
                    keypoints_str = "\n* ".join(result_keypoints).lstrip()
                else:
                    keypoints_str += "\n* ".join(result_keypoints).lstrip()

                if self.generate_bool("Is the information enough to explain the user's question?", save = False):
                    return keypoints

                if self.generate_bool(f"Is the information about keyword:{keyword} enough?", save = False):
                    break

        return None if keypoints == {} else keypoints

    def summarize(self, analyze_result: dict[googlesearch.SearchResult, list[str]]) -> str:
        messages: list[dict[str, str]] = []
        instruction: str

        if analyze_result is None:
            instruction = "Write the explanation without uncertain things in markdown format."
        else:
            analyze_result_str: str = ""
            for k, p in analyze_result.items():
                for pi in p:
                    analyze_result_str += f"{pi}\n"
            messages.append({ "role": "system", "content": analyze_result_str })
            instruction = "Write the explanation from gathered information in markdown format."

        return self.generate_string_freely(instruction, messages = messages)

    def update_qa_response(self, prompt: str, result: str):
        response: list[dict[str, str]] = [
            { "role": "user", "content": prompt },
            { "role": "assistant", "content": result }
        ]

        self.qa_response += response
        self.logger.debug(self.qa_response)
        self.messages = self.messages[:1] + self.qa_response
        self.logger.debug(self.messages)

    def verify_xml(self, root_tag: str, children: dict[str, Any], verify: Callable[[ET.Element], bool] = lambda x: True) -> Callable[[str], bool]:
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
                    if not k.startswith('<') and not k.endswith('>'): # If it is not a tag
                        continue

                    k_items: list[ET.Element] = sub.findall(k.strip("<>"))
                    if k_items is None:
                        continue

                    n_tags += len(k_items)
                    
                    if type(v) == dict:
                        for k_item in k_items:
                            n_tags += __verify_sub(k_item, v)
                return n_tags

            return __verify_sub(xml, children) == __num_tags(xml) and verify(xml) if verify is not None else True
        return __verify

    def xml_example(self, examples: str | list[str], root_tag: str, children: dict[str, str | dict[str, Any]]) -> str:
        output: str = "You have to use XML format to write your response. Examples:\n"
        for i, ex in enumerate(examples if type(examples) == list else [examples]):
            output += f"Example {i}\n{ex}\n\n"

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

        if children is None:
            output += "No tags other than the root tag are admitted."
        else:
            output += "Valid tags:\n" + __expand(children) + "\nNOTE: No tags other than above are admitted."

        return output

    def xml_instruction_and_verify(self, examples: str | list[str], root_tag: str, children: dict[str, Any], verify: Callable[[ET.Element], bool] = lambda x: True) -> (str, Callable[[str], bool]):
        instruction: str = self.xml_example(examples, root_tag, children)
        verify: Callable[[str], bool] = self.verify_xml(root_tag, children, verify)
        return instruction, verify

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default = "warning", help = "Log level: [debug, info, warning error, critical]")
    exit(Application().run(parser.parse_args()))
