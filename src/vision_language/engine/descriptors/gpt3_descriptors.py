import json
import os
import re


def load_json(filename):
    """Load a JSON file, adding a .json extension if needed."""
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as f:
        return json.load(f)

def wordify(text):
    """Replace underscores with spaces in a class name."""
    return text.replace('_', ' ')

def make_sentence(desc):
    """
    Modify a descriptor into a sentence.
    If the descriptor starts with certain words, format accordingly.
    """
    if desc.startswith(('a ', 'an ')):
        return f"which is {desc}"
    elif desc.startswith(('has', 'often', 'typically', 'may', 'can')):
        return f"which {desc}"
    elif desc.startswith('used'):
        return f"which is {desc}"
    else:
        return f"which has {desc}"

def modify_descriptor(desc, apply_changes):
    """Apply changes to the descriptor if requested."""
    return make_sentence(desc) if apply_changes else desc

def process_name(classname, dsname):
    if dsname == 'stanford_cars':
        names = classname.split(" ")     # ['Chevrolet', 'Impala', '2007']
        year = names.pop(-1)             # year = '2007'; names = ['Chevrolet', 'Impala']
        names.insert(0, year)            # names = ['2007', 'Chevrolet', 'Impala']
        return " ".join(names)     # '2007 Chevrolet Impala'
    elif dsname == 'sun397':
        match = re.match(r"(.+?)\s*\((.+?)\)", classname) # convert 'indoor path (interior)' to 'interior indoor_path'
        if match:
            base = match.group(1).strip().replace(" ", "_")
            tag = match.group(2).strip()
            return f"{tag} {base}"
        else:
            return classname.replace(" ", "_")  # fallback if no parentheses
    return classname

def load_gpt_descriptions(hparams):
    """
    Load GPT-generated descriptions from a JSON file specified in hparams.
    
    Parameters:
      - hparams: a dictionary containing:
            'fname': path to the JSON file,
            'position_class': one of {None, 'append', 'prepend'},
            'modify': bool,
            'between_text': string to insert between descriptor and class name,
            'before_text': text to prepend if using 'prepend',
            'after_text': text to append if using 'prepend'.
    
    Returns:
      - descriptions: a dict mapping each class to a list of modified descriptor strings.
      - unmodified: a dict mapping each class to a mapping {modified: original} for reference.
    """

    descriptions = load_json(hparams['fname'])
    combined = {}
    unmodified = {}
    out = {}
    
    # Only modify descriptors if a category inclusion mode is specified.
    for i, (cls, desc_list) in enumerate(descriptions.items()):
        
        desc_list = [''] if not desc_list else desc_list # Ensure we have at least one descriptor.
        cls_p = process_name(cls, hparams['dsname'])
        cls_name = wordify(cls)
        
        inclusion = hparams['position_class']
        if inclusion == 'append':
            build = lambda item: f"{modify_descriptor(item, hparams['modify'])}{hparams['between_text']}{cls_name}"
        elif inclusion == 'prepend':
            build = lambda item: f"{hparams['before_text']}{cls_name}{hparams['between_text']}{modify_descriptor(item, hparams['modify'])}{hparams['after_text']}"
        else:
            build = lambda item: modify_descriptor(item, hparams['modify'])
        
        # Build mapping for this class.
        unmodified[cls_p] = {build(item): item for item in desc_list}
        if hparams.get('combine'):
            out[cls_p] = f"{cls_name}: {', '.join(desc_list)}"
        else:
            out[cls_p] = [build(item) for item in desc_list]
    return out, unmodified


if __name__ == '__main__':
    # Example usage.
    file = os.path.join('./descriptions', f"descriptors_food101.json")
    hparams = {'position_class': 'prepend',
        'modify': True,
        'before_text': '',
        'between_text': ', ',
        'after_text': '',
        'combine': True}
    hparams['fname'] = file
    descriptions, combined = load_gpt_descriptions(hparams)
    # print(descriptions)