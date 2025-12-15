import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from common.string_normalization import contains, remove_snake_case, apply_label_casing, apply_single_quotes, \
    replace_x_and_y
from common.vectorization import vector_top_k
from util.logger_config import config

logger = logging.getLogger(__name__)

config(logger)


def generate_definition_prefix(elem_definition: str, elem_label: str, elem_type: str) -> str:
    if not contains(elem_definition, elem_label):
        if elem_type == "class":
            elem_definition = f"individual x is a '{elem_label}' iff individual x is a {elem_definition}"
        if elem_type == "property":
            elem_definition = f"individual x '{elem_label}' individual y iff {elem_label} is a {elem_definition}"
    return elem_definition


def create_automatic_class_definition(elem_label: str, phrase_diffs_dict, target_info: dict[str, Any]) -> str:
    axiomatic_definition = f"individual x is a '{elem_label}' iff"
    if target_info.get("parent_label"):
        axiomatic_definition += f" individual x is a '{target_info.get('parent_label')}' "
    if phrase_diffs_dict.get(elem_label) and str(phrase_diffs_dict[elem_label]) != "nan":
        axiomatic_definition += f", such that {phrase_diffs_dict[elem_label]}"
    else:
        axiomatic_definition += ", such that no other axioms hold"
    axiomatic_definition += "."
    return axiomatic_definition


def create_automatic_property_definition(elem_label: str, phrase_diffs_dict, target_info: dict[str, Any]) -> str:
    parent_label = target_info.get("parent_label")
    child_domain = target_info.get("child_domain")
    child_range = target_info.get("child_range")

    axiomatic_definition = f"individual x '{elem_label}' individual y iff"
    if target_info.get("parent_label"):
        axiomatic_definition += f" individual x '{target_info.get('parent_label')}' individual y "
    if target_info.get("child_domain"):
        axiomatic_definition += f" {'and' if parent_label else ''} individual x is an instance of '{target_info.get('child_domain')}'"
    if target_info.get("child_range"):
        axiomatic_definition += f" {'and' if parent_label or child_domain else ''} individual y is an instance of '{target_info.get('child_range')}'"
    # if phrase_diffs_dict.get(elem_label) and str(phrase_diffs_dict[elem_label]) != "nan":
    #     axiomatic_definition += f" {'and' if parent_label or child_domain or child_range else ''} {phrase_diffs_dict[elem_label]}"
    # else:
    #     axiomatic_definition += ", such that no other axioms hold"
    axiomatic_definition += "."
    return axiomatic_definition


def normalize_definition_prefix(improved_definition: str,
                                type_name: str,
                                automatic_definition: str) -> str:
    if "such that" in improved_definition:
        automatic_definition = automatic_definition.strip('.').strip()
        improved_definition = improved_definition.split("such that")[1].strip()
        improved_definition = f"{automatic_definition}, such that {improved_definition}"
    return improved_definition


def clean_up_definition(elem_definition: str, elem_label: str, elem_type: str, ref_labels) -> str:
    elem_definition = remove_snake_case(elem_definition)
    elem_definition = apply_label_casing(elem_definition, ref_labels)
    elem_definition = apply_single_quotes(elem_definition, ref_labels)
    elem_definition = replace_x_and_y(elem_definition)
    elem_definition = generate_definition_prefix(elem_definition, elem_label, elem_type)
    return elem_definition


def create_class_definition_prompt(elem_iri: str,
                                   elem_label: str,
                                   elem_type: str,
                                   elem_definition: str,
                                   automatic_definition: str,
                                   target_info,
                                   prompts, phrase_diffs_dict,
                                   ref_entries,
                                   ref_labels,
                                   settings: dict) -> ChatPromptTemplate:
    system_prompt = prompts.get("class").get("system")
    user_prompt = prompts.get("class").get("user")

    ###################################################################################
    # Clean up the definition
    elem_definition = clean_up_definition(elem_definition, elem_label, elem_type, ref_labels)

    user_prompt = user_prompt.replace("{label}", elem_label)
    user_prompt = user_prompt.replace("{definition}", elem_definition)

    ##################################################################################
    # Insert automatic definition
    user_prompt = user_prompt.replace("{automatic_definition}", automatic_definition)

    ##################################################################################
    # Create reference contexts
    query = f"{elem_label or ''} {elem_definition or ''}"
    ref_top_k = settings.get("ref_top_k")

    # Perform vector search for property and class references
    property_references = vector_top_k(query, "property", ref_top_k, settings)
    class_references = vector_top_k(query, "class", ref_top_k, settings)

    # Add property and class references to prompt
    property_reference_lines = []
    for ref in property_references:
        property_reference_lines.append(f"- {ref['label']} {ref['definition']}")
    property_reference_context = "\n".join(property_reference_lines).strip('\n')
    user_prompt = user_prompt.replace("{property_reference_context}", property_reference_context)

    class_reference_lines = []
    for ref in class_references:
        class_reference_lines.append(f"- {ref['label']} {ref['definition']}")
    class_reference_context = "\n".join(class_reference_lines).strip('\n')
    user_prompt = user_prompt.replace("{class_reference_context}", class_reference_context)

    ollama_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_prompt.strip()),
        ]
    )
    return ollama_prompt


def create_property_definition_prompt(elem_iri: str,
                                      elem_label: str,
                                      elem_type: str,
                                      elem_definition: str,
                                      automatic_definition: str,
                                      target_info,
                                      prompts, phrase_diffs_dict,
                                      ref_entries,
                                      ref_labels,
                                      settings: dict) -> ChatPromptTemplate:
    system_prompt = prompts.get("property").get("system")
    user_prompt = prompts.get("property").get("user")

    ###################################################################################
    # Clean up the definition
    elem_definition = clean_up_definition(elem_definition, elem_label, elem_type, ref_labels)

    user_prompt = user_prompt.replace("{label}", elem_label)
    user_prompt = user_prompt.replace("{definition}", elem_definition)

    ##################################################################################
    # Create automatic definition
    user_prompt = user_prompt.replace("{automatic_definition}", automatic_definition)

    lines = []
    if target_info.get("parent_label"):
        lines.append(f"The parent property 'Y' of '{elem_label}' is: '{target_info.get('parent_label')}'")
        if target_info.get("parent_domain"):
            lines.append(
                f"The domain 'C' of the parent property 'Y' is: '{target_info.get('parent_domain')}'")
        else:
            lines.append("This parent property 'Y' has no domain 'C'.")

        if target_info.get("parent_range"):
            lines.append(f"The range 'D' of the parent property 'Y' is: '{target_info.get('parent_range')}'")
        else:
            lines.append("The parent property 'Y' has no range 'D'.")
    else:
        lines.append("This property has no parent property.")
    parent_context = "\n".join(lines).strip('\n')
    user_prompt = user_prompt.replace("{parent_context}", parent_context)

    ##################################################################################
    # Create reference contexts
    query = f"{elem_label or ''} {elem_definition or ''}"
    ref_top_k = settings.get("ref_top_k")

    # Perform vector search for property and class references
    property_references = vector_top_k(query, "property", ref_top_k, settings)
    class_references = vector_top_k(query, "class", ref_top_k, settings)

    # Add property and class references to prompt
    property_reference_lines = []
    for ref in property_references:
        property_reference_lines.append(f"- {ref['label']} {ref['definition']}")
    property_reference_context = "\n".join(property_reference_lines).strip('\n')
    user_prompt = user_prompt.replace("{property_reference_context}", property_reference_context)

    class_reference_lines = []
    for ref in class_references:
        class_reference_lines.append(f"- {ref['label']} {ref['definition']}")
    class_reference_context = "\n".join(class_reference_lines).strip('\n')
    user_prompt = user_prompt.replace("{class_reference_context}", class_reference_context)

    ollama_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_prompt.strip()),
        ]
    )
    return ollama_prompt
