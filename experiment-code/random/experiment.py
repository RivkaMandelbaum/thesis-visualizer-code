# pylint: disable=unused-import,abstract-method,unused-argument

##########################################################################################
# Imports
##########################################################################################

import random
from typing import List, Optional

import numpy as np
from flask import Markup
from scipy import stats
import networkx as nx

import psynet.experiment
from psynet.consent import PrincetonConsent
from psynet.graphics import Circle, Frame, GraphicPrompt
from psynet.modular_page import ModularPage, NullControl, Prompt, PushButtonControl, NumberControl, TextControl
from psynet.page import InfoPage, SuccessfulEndPage
from psynet.timeline import Timeline, FailedValidation, join, CodeBlock
from psynet.trial.graph import (
    GraphChainNetwork,
    GraphChainNode,
    GraphChainSource,
    GraphChainTrial,
    GraphChainTrialMaker,
)
from psynet.prescreen import ColorBlindnessTest
from psynet.utils import get_logger

logger = get_logger()


##########################################################################################
# Stimuli
##########################################################################################
COLOR_OPTIONS = ["red", "green", "blue"]
NECKLACE_LENGTH = 9

##########################################################################################
# Constants
##########################################################################################

INITIAL_RECRUITMENT_SIZE = 20
NUM_ITERATIONS_PER_CHAIN = 20
NUM_TRIALS_PER_PARTICIPANT = 30
RANDOM_DEGREE = 4
NUMBER_OF_NODES = 49

instructions = join(
    InfoPage(
        Markup(
            """
            <p>
                In this experiment you will observe a selection of colorful necklaces made of beads that can
                be either red, green or blue. In each round, you will be asked to choose a necklace that you like
                most and then attempt to reproduce it from memory. You will do that by coloring a new necklace
                by clicking on its beads so that their color matches the one you selected.
            </p>
            """
        ),
        time_estimate=5
    ),
    InfoPage(
        Markup(
            """
            <p>
                In certain cases you will have only one necklace to choose, in which case you have to reproduce it from memory.
            </p>
            """
        ),
        time_estimate=5
    ),
    InfoPage(
        Markup(
            """
            <p>
                We are specifically interested in seeing how people <strong>intuitively</strong> choose and reproduce necklaces,
                not in perfect accuracy! The best strategy is to just pay attention and answer what you intuitively think
                is the right answer, no prior expertise is required to complete this task.
            </p>
            """
        ),
        time_estimate=5
    ),
    InfoPage(f"You will complete up to {NUM_TRIALS_PER_PARTICIPANT} rounds, where each round presents a new set of necklaces.", time_estimate=3)
)

demographics = join(
    InfoPage("Before we finish, we need to ask some quick questions about you.", time_estimate=5),
    ModularPage(
        "age",
        Prompt("Please indicate your age."),
        control=NumberControl(),
        time_estimate=3,
    ),
    ModularPage(
        "gender",
        Prompt("What gender do you identify as?"),
        control=PushButtonControl(
            ["Female", "Male", "Other"], arrange_vertically=False
        ),
        time_estimate=3,
    ),
    ModularPage(
        "education",
        Prompt("What is your highest educational qualification?"),
        control=PushButtonControl(
            [
                "None",
                "Elementary school",
                "Middle school",
                "High school",
                "Undergraduate",
                "Graduate",
                "PhD",
            ],
            arrange_vertically=False,
        ),
        time_estimate=3,
    ),
    ModularPage(
        "country",
        Prompt(
            """
            What country are you from?
            """
        ),
        control=TextControl(one_line=True),
        time_estimate=3,
    ),
    ModularPage(
        "mother_tongue",
        Prompt(
            """
            What is your first language?
            """
        ),
        control=TextControl(one_line=True),
        time_estimate=3,
    ),
)

final_questionnaire = join(
    ModularPage(
        "strategy",
        Prompt(
            """
        Please tell us in a few words about your experience taking the task.
        What was your strategy?
        Did you find the task easy or difficult?
        Did you find it interesting or boring?
        """
        ),
        control=TextControl(one_line=False),
        time_estimate=10,
    ),
    ModularPage(
        "technical",
        Prompt(
            """
        Did you experience any technical problems during the task?
        If so, please describe them.
        """
        ),
        control=TextControl(one_line=False),
        time_estimate=10,
    ),
)

class NecklaceCircle(Circle):
    """
    A Necklace circle object.

    Parameters
    ----------

    id_
        A unique identifier for the object.

    x
        x coordinate.

    y
        y coordinate.

    radius
        The circle's radius.

    color_options
        Color options (as a list of strings) for the circles

    interactive
        boolean

    initial_color
        index into color_options array for initial color of circle, or -1
        for gray

    color_on_click
        index into color_options array for the color a circle turns after
        it is clicked on the first time (default: None)

    **kwargs
        Additional parameters passed to :class:`~psynet.graphic.GraphicObject`.
    """

    def __init__(
        self,
        id_: str,
        x: int,
        y: int,
        radius: int,
        color_options: List[str],
        initial_color: int,
        interactive: bool,
        color_on_click=None,
        **kwargs,
    ):
        self.color_options = color_options
        self.initial_color = initial_color
        self.color_on_click = color_on_click
        self.interactive = interactive
        super().__init__(id_, x, y, radius, click_to_answer=not interactive, **kwargs)

    @property
    def js_init(self) -> str:
        return [
            *super().js_init,
            f"""
            let initial_color = {self.initial_color};
            let color_options = {self.color_options};
            if (initial_color == -1) {{
                this.raphael.attr({{"stroke": "gray", "fill": "gray"}});
            }}
            else {{
                this.raphael.attr({{"stroke": color_options[initial_color], "fill": color_options[initial_color]}});
            }}

            let color_on_click = null;
            if ("{self.color_on_click != None}" == "True") {{
                color_on_click = {self.color_on_click};
            }}

            if (psynet.response.staged.rawAnswer == undefined) {{
                psynet.response.staged.rawAnswer = {{}};
            }}

            let stage_color = function(index, circle_id) {{
                psynet.response.staged.rawAnswer[circle_id] = {{
                    color_index: index,
                    color_value: color_options[index]
                }};
            }};

            stage_color(initial_color, "{self.id}");

            this.raphael.click(function () {{
                if ("{self.interactive}" == "True") {{
                    let currentColor = this.attrs.fill;
                    let targetIdx = (color_options.findIndex(element => element == currentColor) + 1) % color_options.length
                    if (color_on_click != null) {{
                        targetIdx = {self.color_on_click};
                        color_on_click = null;
                    }}
                    this.attr({{"stroke": color_options[targetIdx], "fill": color_options[targetIdx]}});
                    stage_color(targetIdx, "{self.id}");
                }}
            }});
            """,
        ]


class CustomGraphicPrompt(GraphicPrompt):
    def validate(self):
        return True

class NecklaceControl(NullControl):
    def validate(self, response, **kwargs):
        for color_index in response.answer:
            if color_index < 0:
                return FailedValidation("You must color all circles in the necklace to continue!")
        return None

class NecklaceNAFCPage(ModularPage):
    def __init__(
        self,
        label: str,
        prompt: str,
        necklace_states: List[List[int]],
        color_options: List[str],
        time_estimate=10,
        dimensions = [640,480],
        css="",
        scripts="",
    ):
        self.color_options = color_options
        self.necklace_states = necklace_states

        py = 100
        if len(necklace_states) == 1:
            py = .5 * dimensions[1]

        activate_submit_without_click = True

        alert_mac_screenshot = 'let cmdShiftAlert = event => { if (event.shiftKey && event.metaKey) { alert("We have detected that you may be trying to take a screenshot. Screenshots of this experiment are prohibited.") } }; document.addEventListener("keydown", cmdShiftAlert);'

        alert_mac_windows_printkeys = 'const macPrintEvent = event => { if (event.key === "p" && (event.metaKey || event.ctrlKey)) { alert("We have detected that you are trying to print the page. Printing this page is prohibited."); } }; document.addEventListener("keydown", macPrintEvent)'

        alert_mac_windows_savekeys = 'const macSaveEvent = event => { if (event.key === "s" && (event.metaKey || event.ctrlKey)) { alert("We have detected that you are trying to save the page. Downloading this page is prohibited."); } }; document.addEventListener("keydown", macSaveEvent)'

        super().__init__(
            label,
            prompt=CustomGraphicPrompt(
                text=prompt,
                dimensions=dimensions,
                viewport_width=0.7,
                frames=[
                    Frame(
                        self.create_necklace_array(
                            px=150,
                            py=py,
                            size=20,
                            spacing=41,
                            vertical_spacing=75,
                            necklace_states=necklace_states,
                            color_options=color_options,
                            interactive=False, # click to answer = not interactive
                        ),
                        activate_control_submit=False,
                    )
                ],
                prevent_control_submit=activate_submit_without_click,
            ),
            time_estimate=time_estimate,
            css=css,
            scripts=[alert_mac_screenshot, alert_mac_windows_printkeys, alert_mac_windows_savekeys],
        )

    def format_answer(self, raw_answer, **kwargs):
        if "clicked_object" not in raw_answer:
            chosen_necklace_id = 0
        else:
            chosen_necklace_id = int(raw_answer["clicked_object"].split("_")[1])
        return chosen_necklace_id

    def create_necklace(
        self, px, py, size, spacing, coloring, color_options, necklace_id, interactive
    ):
        translation = 0
        necklace = []
        for i in range(len(coloring)):
            necklace = necklace + [
                NecklaceCircle(
                    id_=necklace_id + "_circle_" + str(i),
                    x=px + translation,
                    y=py,
                    radius=size,
                    color_options=color_options,
                    initial_color=coloring[i],
                    interactive=interactive,
                    attributes='{"style": @media print { html, #graphic-prompt-container { display: none; } }}',
                )
            ]
            translation += spacing
        return necklace

    def create_necklace_array(
        self, necklace_states, vertical_spacing, px, py, **kwargs
    ):
        translation = 0
        necklace_array = []
        for i in range(len(necklace_states)):
            necklace_array = necklace_array + self.create_necklace(
                necklace_id="necklace_" + str(i),
                px=px,
                py=py + translation,
                coloring=necklace_states[i],
                **kwargs,
            )
            translation += vertical_spacing
        return necklace_array


class NecklaceInteractivePage(ModularPage):
    def __init__(
        self,
        label: str,
        prompt: str,
        necklace_state: List[List[int]],
        color_options: List[str],
        time_estimate=10,
        dimensions=[640,480],
        initial_color=None,
        color_on_click=None,
    ):
        self.color_options = color_options
        self.necklace_state = necklace_state
        self.initial_color = initial_color
        self.color_on_click = color_on_click

        super().__init__(
            label=label,
            prompt=GraphicPrompt(
                text=prompt,
                dimensions=dimensions,
                viewport_width=0.7,
                frames=[
                    Frame(
                        self.create_necklace(
                            necklace_id="necklace",
                            px=140,
                            py= (.5 * dimensions[1]),
                            size=20,
                            spacing=41,
                            coloring=necklace_state,
                            color_options=color_options,
                            interactive=True,
                            initial_color=initial_color,
                            color_on_click=color_on_click
                        )
                    )
                ],
            ),
            control=NecklaceControl(),
            time_estimate=time_estimate,
        )

    def format_answer(self, raw_answer, **kwargs):
        chosen_state = [None for _ in range(len(raw_answer.keys()))]
        for key in raw_answer.keys():
            idx = int(key.split("_")[2])
            chosen_state[idx] = raw_answer[key]["color_index"]
        return chosen_state

    def create_necklace(
        self, px, py, size, spacing, coloring, color_options, necklace_id, interactive, initial_color, color_on_click
    ):
        translation = 0
        necklace = []

        if initial_color == None:
            color_on_click = None
        else:
            if color_on_click == None:
                color_on_click = random.randint(0, len(COLOR_OPTIONS) - 1)
            # if it's already been assigned, don't overwrite it

        for i in range(len(coloring)):
            if (initial_color == None):
                initial_color = coloring[i]
            necklace = necklace + [
                NecklaceCircle(
                    id_=necklace_id + "_circle_" + str(i),
                    x=px + translation,
                    y=py,
                    radius=size,
                    color_options=color_options,
                    initial_color=initial_color,
                    color_on_click=color_on_click,
                    interactive=interactive,
                )
            ]
            translation += spacing
        return necklace


class CustomTrial(GraphChainTrial):
    accumulate_answers = True
    time_estimate = 20

    def show_trial(self, experiment, participant):
        options = [option["content"] for option in self.definition]


        click_prompt = "Click on the necklace which you like most."

        page_1 = NecklaceNAFCPage(
            label="choose",
            prompt=click_prompt,
            necklace_states=options,
            color_options=COLOR_OPTIONS,
            css=["@media print { #graphic-prompt-container { display:none !important}}"], # disable printing the necklaces
        )

        page_2 = NecklaceInteractivePage(
            label="reproduce",
            prompt="Recolor this necklace like the necklace you just chose.",
            necklace_state=CustomSource.generate_class_seed(),
            color_options=COLOR_OPTIONS,
            initial_color=-1,
            color_on_click=participant.var.color_on_click,
        )

        print("PARTICIPANT COLOR: %s" % COLOR_OPTIONS[participant.var.color_on_click])

        return [page_1, page_2]


class CustomNetwork(GraphChainNetwork):
    pass


class CustomNode(GraphChainNode):
    def summarize_trials(self, trials: list, experiment, paricipant):
        answers = np.array([trial.answer[1] for trial in trials])
        summary = stats.mode(answers)
        return summary.mode.flatten().tolist()

    def get_parents(self): # patch broken method, from https://gitlab.com/computational-audition-lab/singing-networks/ising-graph/-/blob/main/experiment.py
        trial_maker_id = self.network.trial_maker_id
        degree = self.degree
        nodes = GraphChainNode.query.all()
        current_layer = [
            n
            for n in nodes
            if n.network.trial_maker_id == trial_maker_id and n.degree == degree and not n.failed
        ]
        parents = [n for n in current_layer if n.vertex_id in self.dependent_vertex_ids]
        return parents



class CustomSource(GraphChainSource):
    @staticmethod
    def generate_class_seed():
        return [
            random.randint(0, len(COLOR_OPTIONS) - 1) for i in range(NECKLACE_LENGTH)
        ]


class CustomTrialMaker(GraphChainTrialMaker):
    """
        This TrialMaker implements a random d-regular of size number_of_nodes and d = graph_degree.
        """

    response_timeout_sec = 180
    check_timeout_interval_sec = 60

    def __init__(
            self,
            *,
            id_,
            network_class,
            node_class,
            source_class,
            trial_class,
            phase: str,
            graph_degree: int,
            number_of_nodes: int,
            chain_type: str,
            num_trials_per_participant: int,
            num_chains_per_participant: Optional[int],
            trials_per_node: int,
            balance_across_chains: bool,
            check_performance_at_end: bool,
            check_performance_every_trial: bool,
            recruit_mode: str,
            target_num_participants=Optional[int],
            num_iterations_per_chain: Optional[int] = None,
            num_nodes_per_chain: Optional[int] = None,
            fail_trials_on_premature_exit: bool = False,
            fail_trials_on_participant_performance_check: bool = False,
            propagate_failure: bool = True,
            num_repeat_trials: int = 0,
            wait_for_networks: bool = False,
            allow_revisiting_networks_in_across_chains: bool = False,
            graph_seed: Optional[int] = None
    ):
        network_structure = self.generate_graph(graph_degree, number_of_nodes, graph_seed)
        super().__init__(
            id_=id_,
            network_class=network_class,
            node_class=node_class,
            source_class=source_class,
            trial_class=trial_class,
            phase=phase,
            network_structure=network_structure,
            chain_type=chain_type,
            num_trials_per_participant=num_trials_per_participant,
            num_chains_per_participant=num_chains_per_participant,
            trials_per_node=trials_per_node,
            balance_across_chains=balance_across_chains,
            check_performance_at_end=check_performance_at_end,
            check_performance_every_trial=check_performance_every_trial,
            recruit_mode=recruit_mode,
            target_num_participants=target_num_participants,
            num_iterations_per_chain=num_iterations_per_chain,
            num_nodes_per_chain=num_nodes_per_chain,
            fail_trials_on_premature_exit=fail_trials_on_premature_exit,
            fail_trials_on_participant_performance_check=fail_trials_on_participant_performance_check,
            propagate_failure=propagate_failure,
            num_repeat_trials=num_repeat_trials,
            wait_for_networks=wait_for_networks,
            allow_revisiting_networks_in_across_chains=allow_revisiting_networks_in_across_chains,
        )

    def generate_graph(self, d, n, seed):
        G = nx.random_regular_graph(d, n, seed=seed)
        return make_psynet_compatible(G)

def make_psynet_compatible(G, is_directed=False):
        vertices = [n for n in G.nodes]
        edges = []
        for e in G.edges:

            edges.append({
                "origin": e[0],
                "target": e[1],
                "properties": {"type": "default"}
            })

            if not is_directed:
                edges.append({
                    "origin": e[1],
                    "target": e[0],
                    "properties": {"type": "default"}
                })
        return {"vertices": vertices, "edges": edges}



##########################################################################################
# Experiment
##########################################################################################


# Weird bug: if you instead import Experiment from psynet.experiment,
# Dallinger won't allow you to override the bonus method
# (or at least you can override it but it won't work).
class Exp(psynet.experiment.Experiment):
    variables = {
        "wage_per_hour": 12.0,
        "show_bonus": False,
        "soft_max_experiment_payment": 2000.0,
        "hard_max_experiment_payment": 2500.0
    }
    timeline = Timeline(
        PrincetonConsent(),
        ColorBlindnessTest(),
        instructions,
        InfoPage("To help familiarize you with the task, you will see an example necklace now.", time_estimate=5),
        NecklaceInteractivePage(
            label="reproduce_example",
            prompt=("You can recolor this necklace by clicking on its beads. Each time you click on a bead, the color will change. Once you're done you can click Next."),
            necklace_state=CustomSource.generate_class_seed(),
            color_options=COLOR_OPTIONS,
            initial_color=-1,
        ),
        CodeBlock(lambda participant: participant.var.set("color_on_click", random.randint(0, len(COLOR_OPTIONS)-1))),
        InfoPage("The experiment will begin now!", time_estimate=3),
        CustomTrialMaker(
            id_="graph_experiment",
            network_class=CustomNetwork,
            trial_class=CustomTrial,
            node_class=CustomNode,
            source_class=CustomSource,
            phase="experiment",
            chain_type="across",
            num_iterations_per_chain=NUM_ITERATIONS_PER_CHAIN,
            num_trials_per_participant=NUM_TRIALS_PER_PARTICIPANT,
            num_chains_per_participant=None,
            trials_per_node=1,
            balance_across_chains=True,
            check_performance_at_end=False,
            check_performance_every_trial=False,
            recruit_mode="num_trials",
            target_num_participants=None,
            graph_degree=RANDOM_DEGREE,
            number_of_nodes=NUMBER_OF_NODES,
        ),
        InfoPage("You finished the experiment!", time_estimate=3),
        demographics,
        final_questionnaire,
        InfoPage(Markup(
            """
            <strong>Attention</strong>: If you experience any problems in submitting the HIT, don't worry, just send us
            an email <strong>directly at cocosci.turk+rivka@gmail.com</strong> with your accurate worker id and bonus.
            Please avoid using the automatic error boxes. This will help us compensate you appropriately.
            """
        ),time_estimate=5),
        SuccessfulEndPage(),
    )

    def __init__(self, session=None):
        super().__init__(session)
        self.initial_recruitment_size = INITIAL_RECRUITMENT_SIZE
