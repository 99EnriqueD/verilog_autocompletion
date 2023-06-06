import re
import pandas as pd
import numpy as np
import ast
import random
from bs4 import BeautifulSoup

# https://stackoverflow.com/questions/16198546/get-exit-code-and-stderr-from-subprocess-call
# https://stackoverflow.com/questions/5596911/python-os-system-without-the-output
import subprocess
from subprocess import DEVNULL, STDOUT, check_call, check_output 

import uuid
import pyverilog
from pyverilog.vparser.parser import parse
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator

### PANDAS

def read_csv(path):
    df = pd.read_csv(path,index_col=0)
    df = df.set_index(df.index.astype(str))
    return df.fillna("")

def list_is_not_empty(row,col_name):
    try:
        return len(ast.literal_eval(row[col_name])) > 0
    except Exception as e:
        # print(f"Could not parse string of length {len(row[col_name])}: {row[col_name]}")
        # print(f"ERROR::{e}")
        return len(row[col_name]) > 2

def count_keywords(df,col_name):
    keyword_count = {}
    for i, row in df.iterrows():
        try:
            kw_list = ast.literal_eval(row[col_name])
            for kw in kw_list:
                if kw in keyword_count:
                    keyword_count[kw] += 1
                else:
                    keyword_count[kw] = 1
        except Exception as e:
            # print(f"ERROR::{e}")
            ...
    return keyword_count
        
def kw_list_contains(row, col_name, kw):
    try:
        kw_list = ast.literal_eval(row[col_name])
        return kw in kw_list
    except Exception as e:
        # print(f"Could not parse string of length {len(row[col_name])}: {row[col_name]}")
        # print(f"ERROR::{e}")
        return False
        
def filter_df_by_license_keywords(df):
    # Filter all rows that have non-empty keyword blacklist and empty keyword whitelist
    blacklist_not_empty = df.apply(lambda x: list_is_not_empty(x, 'license_blacklist_keywords'), axis=1)
    whitelist_not_empty = df.apply(lambda x: list_is_not_empty(x, 'license_whitelist_keywords'), axis=1)
    return df[np.logical_or(np.logical_not(blacklist_not_empty),whitelist_not_empty)]

def filter_df_if_index_not_in_set(df,set):
    return df[df.apply]

### STRING PROCESSING
# Find comments

multi_line_comment_re = re.compile(
    r'(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*)($)?',
    re.DOTALL | re.MULTILINE
)
single_line_comment_re = re.compile(
    r'(^)?[^\S\n]*(//[^\n]*)($)?',
    re.DOTALL | re.MULTILINE
)

# string_subpattern_1 = r'".*"'
# string_subpattern_2 = r"'.*'"
# string_pattern_re = re.compile(f"{string_subpattern_1}|{string_subpattern_2}")
string_pattern_re = re.compile(r'"[^"\\]*(?:\\[\s\S][^"\\]*)*"')

def remove_multi_line_comments(string):
    return re.sub(multi_line_comment_re, "", string)
def remove_single_line_comments(string):
    return re.sub(single_line_comment_re, "", string) 
def remove_all_comments(string):
    return remove_single_line_comments(remove_multi_line_comments(string)).strip()

def get_multi_line_comments(string):
    return re.findall(multi_line_comment_re, string)
def get_single_line_comments(string):
    return re.findall(single_line_comment_re, string)

def get_pattern_indices(pattern,string):
    return [m for m in re.finditer(pattern, string)]

def abstract_strings(inp):
    ms = get_pattern_indices(string_pattern_re,inp)
    replaced = []
    count = 0
    index_reduce = 0
    for m in ms:
        replaced.append(inp[m.start() + index_reduce:m.end()+ index_reduce])
        new_inp = '"' + str(count) + '"'
        count += 1
        inp = inp[:m.start() + index_reduce] + new_inp + inp[m.end() + index_reduce:]
        index_reduce += len(new_inp) - (m.end() - m.start())
    return inp, replaced

def recreate_string(inp,replaced):
    
    def replace(match):
        repl_int = int(match.string[match.start()+1:match.end()-1])
        return replaced[repl_int]
        
    return re.sub(r'"\d+"',replace,inp)


# Process comments

def remove_forward_slashes(string):
    return string.replace("/","")
def condense_spaces(string):
    return re.sub("\s+", " ", string).strip()

def condense_newlines(string):
    return re.sub("\n+", "\n",string).strip()

def process_found_comments(findall_matches):
    out = ""
    for match in findall_matches:
        processed_match = condense_spaces(remove_forward_slashes(match[1]))
        out += " " + processed_match
    return condense_spaces(out).strip()

# Find without string/comment
def find_first_code_substring(string,substring):
    first_index = string.find(substring)
    for pattern in [string_pattern_re, single_line_comment_re, multi_line_comment_re]:
        matches = re.finditer(pattern, string)
        for match in matches:
            if match and first_index in range(match.span()[0],match.span()[1]):
                if match.span()[1] < len(string):
                    print(f"ffcs updated: found in", match.span()[1])
                    return match.span()[1] + find_first_code_substring(string[match.span()[1]:],substring)
    return first_index

def get_lineno_of_string_index(str_idx,full_string):
    print("====IN GETTING LINE FROM SIDX====", str_idx)
    lines = full_string.split("\n")
    chars_remaining = str_idx
    for i,l in enumerate(lines):
        print(i,chars_remaining,"||",l)
        chars_remaining -= len(l) + 1
        if chars_remaining < 0:
            
            print("====DONE GETTING LINE FROM SIDX====")
            return i
    
    raise Exception("Could not find lineno of string index...")

### PYVERILOG

def print_dir(object):
    print([d for d in dir(object) if not d.startswith("_")])

def generate_code(ast):
    codegen = ASTCodeGenerator()
    return codegen.visit(ast)

# Get level 0 modules (all modules are level 0 probably), and line spans
def mine_module_spans(description_ast):
    children = description_ast.children()
    spans = []
    for child in children:
        if type(child) == pyverilog.vparser.ast.ModuleDef:
            # Get line span...
            spans.append((child.lineno,child.end_lineno))
    return spans

def mine_ast(ast):
    # Mine: modules + functions
    children = ast.children()
    snippets = []
    linenos = []
    for child in children:
        if type(child) in [pyverilog.vparser.ast.ModuleDef,pyverilog.vparser.ast.Function]:
            code = generate_code(child)
            snippets.append(code)
            linenos.append(child.lineno)
        else:
            s, l = mine_ast(child)
            snippets += s
            linenos += l
    return snippets, linenos

def pyverilog_parse_for_row(row):
    exception = "NA"
    row_directory = row['directory'].replace("\\","/")
    file_path = row_directory.rsplit("/",1)[0]
    repo_path = "/".join(row_directory.split("/",4)[:4])
    try:
        ast, _ = parse([row_directory], preprocess_include=[file_path, repo_path], preprocess_define=[])
        # spans = mine_module_spans(ast.description)
        
    except Exception as e:
        snippets,linenos = [],[]
        exception = str(e)
    else:
        snippets, linenos = mine_ast(ast)
        exception = "NA"
    return pd.Series([snippets, linenos, exception])

def synchronize_pyverilog_module_spans(row):
    """
    When file are included (like with: `include 'file.v') lines can be added to the file by pyverilog meaning the module spans given are not right and need to be translated by a few lines
    Apparently include statements can be added mid module...
    """
    
    modules_list = ast.literal_eval(row['icarus_module_spans'])
    lines = row['code'].split("\n")
    modules_list.sort()
    if len(modules_list) < 1:
        return modules_list
    lineno,endlineno = modules_list[0]
    # Find case where module is in middle of module...
    # if lines[lineno-1].strip().startswith("module") and not lines[endlineno-1].strip().endswith("endmodule"):

    if endlineno > len(lines) or not lines[lineno-1].strip().startswith("module") or not lines[endlineno-1].strip().endswith("endmodule"):
        print("+++++++++++++++++++++++")
        print(row['directory'])
        first_index = find_first_code_substring(row['code'],'module')
        new_lineno = get_lineno_of_string_index(first_index,row['code'])
        difference = lineno-1-new_lineno
        print(f'shifting {difference} lines',len(lines))
        # print(f'shifting {difference} lines',len(lines), endlineno > len(lines), not lines[lineno-1].strip().startswith("module"), not lines[endlineno-1].endswith("endmodule"))
        print(modules_list)
        print(row['code'])
        new_module_spans = [(ms[0]-difference,ms[1]-difference) for ms in modules_list]
        print("----------LINES----------")
        print(difference,len(lines))
        print(new_module_spans[0])
        print(lines[new_module_spans[0][0]-1])
        print(lines[new_module_spans[0][1]-1])
        assert(lines[new_module_spans[0][0]-1].strip().startswith('module'))
        assert(lines[new_module_spans[0][1]-1].strip().startswith('endmodule'))
        return new_module_spans
    return modules_list

def get_pyverilog_modules(row):
    # Assumes all rows are lists
    modules_list = ast.literal_eval(row['icarus_module_spans'])
    lines = row['code'].split("\n")
    modules = []
    for lineno, endlineno in modules_list:
        # TODO: check if this is right! Could be endlineno-1
        module = "\n".join(lines[lineno-1:endlineno])
        modules.append(module)
    return modules

def split_pyverilog_module_def_and_body(module_string):
    # Module syntax: https://www.chipverify.com/verilog/verilog-modules
    # Seems to be: body starts after first ";" after module 
    # just be careful about ";" in strings! -> replace with ":" (string will be same length so module itself can )
    module_string = module_string.strip()
    if not module_string.startswith("module") or not module_string.endswith("endmodule"):
        print(module_string)
        raise Exception(f"Module starts//ends with: {module_string[:10]}//{module_string[-10:]}")
    end_def_index = find_first_code_substring(module_string,';')
    def_string = module_string[:end_def_index+1]
    body_string = module_string[end_def_index+1:].strip()
    return def_string, body_string


### VERILATOR
def verilator_parse_for_row(row):
    row_directory = row['directory'].replace("\\","/")
    hanglist = ['data/full_repos/permissive/143278/ivltests/pr3366217h.v']
    if row_directory in hanglist:
        return pd.Series(['NA',"VERILATOR HANGS ON THIS FILE"])
    file_path = row['directory'].rsplit("/",1)[0]
    repo_path = "/".join(row['directory'].split("/",4)[:4])
    out_path = "data/verilator_xmls/" + str(uuid.uuid4()) + ".xml"
    verilator_command = create_verilator_command(row_directory,[file_path, repo_path],out_path)
    # os.system(verilator_command)
    print(row_directory)
    try:
        output = subprocess.check_output(verilator_command.split(), stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        # print("Status : FAIL", exc.returncode, exc.output)
        # print('error')
        return pd.Series(["NA", f"{exc.returncode}: {exc.output}"])
    else:
        # print("SUCCESS")
        return pd.Series([out_path, "NA"])

def create_verilator_command(verilog_file_path, includes, out_path):
    # E.g: verilator --xml-output "./verilator_out.xml" --y ".","./data" test.v
    includes = ",".join(includes)
    return f"verilator --xml-output {out_path} --y {includes} {verilog_file_path}"

def read_xml_data(xml_file_path):
    return BeautifulSoup(open(xml_file_path,'r'),'xml')

def get_module_spans(xml_data):
    # Span = (module_lineno,module_def_endlineno)
    modules = xml_data.find_all('module')
    module_spans = []
    for module in modules:
        loc = module.get('loc')
        module_spans.append((int(loc[1]),int(loc[2])))
    return module_spans

def verilator_module_span_from_row(row):
    xml_data = read_xml_data(row['verilator_xml_output_path'])
    return get_module_spans(get_module_spans(xml_data))


### NEAR-DUPLICATE PROCESSING
def create_near_clone_filter_lookup_table(clones_json):
    lookup_table = {} # file-path: True | False (True == keep this file, False == Remove as duplicate)
    for clone_cluster in clones_json:
        cluster_size = len(clone_cluster)
        assert(cluster_size > 1)
        keep_index = random.randint(0,cluster_size-1)
        for i in range(cluster_size):
            if i == keep_index:
                lookup_table[clone_cluster[i].replace("\\","/")] = True
            else:
                lookup_table[clone_cluster[i].replace("\\","/")] = False
    return lookup_table

def keep_file(file_path,keep_lookup_table):
    proc_file_path = file_path.replace("\\","/")
    if not proc_file_path in keep_lookup_table:
        return True
    else:
        return keep_lookup_table[proc_file_path]

