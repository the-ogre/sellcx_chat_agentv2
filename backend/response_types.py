from pydantic import BaseModel, Field
from typing import List, Dict, Any

class Image(BaseModel):
    url: str = Field(description="URL of the image")
    alt_text: str = Field(description="Alternative text for the image")

class Table(BaseModel):
    headers: List[str] = Field(description="List of column headers for the table")
    rows: List[List[Any]] = Field(description="List of rows, where each row is a list of cell values")

class Button(BaseModel):
    text: str = Field(description="Text displayed on the button")
    action_url: str = Field(description="URL to be opened when the button is clicked")

class DropdownOption(BaseModel):
    label: str = Field(description="Display label of the dropdown option")
    value: str = Field(description="Value of the dropdown option")

class Dropdown(BaseModel):
    options: List[DropdownOption] = Field(description="List of dropdown options")
    default: str = Field(description="Default selected value")

class TextInput(BaseModel):
    placeholder: str = Field(description="Placeholder text for the input field")
    default_value: str = Field(description="Default value of the input field")

class Slider(BaseModel):
    min_value: int = Field(description="Minimum value of the slider")
    max_value: int = Field(description="Maximum value of the slider")
    default_value: int = Field(description="Default value of the slider")

class Checkbox(BaseModel):
    label: str = Field(description="Label for the checkbox")
    checked: bool = Field(description="Whether the checkbox is checked by default")

class RadioButton(BaseModel):
    label: str = Field(description="Label for the radio button")
    value: str = Field(description="Value of the radio button")
    checked: bool = Field(description="Whether the radio button is selected by default")

class UIElement(BaseModel):
    type: str = Field(description="Type of the UI element (e.g., 'button', 'dropdown', 'text_input', 'slider', 'checkbox', 'radio_button')")
    content: Dict[str, Any] = Field(description="Content of the UI element")