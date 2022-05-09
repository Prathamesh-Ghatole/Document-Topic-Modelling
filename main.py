import pickle
import time
from random import random

#importing our custom library.
import model

#######################
# Setting up Rich for pretty printing!

from rich import print
from rich.console import Console
from rich.columns import Columns
from rich.theme import Theme
from rich.pretty import pprint

custom_theme = Theme({
    "info": "dim white",
    "warning": "bold magenta",
    "danger": "bold red",
    "success": "bold green"
})

console = Console(theme=custom_theme)
console.clear()
######################

with open('data/newsgroups', 'rb') as f:
    newsgroup_data = pickle.load(f)

time.sleep(0.3)
console.print("Training Data loaded!\n", style="success")

char_count = 0
for item in newsgroup_data:
    char_count += len(item)

# Using our custom library function to preprocess the data 
# and fetch the main corpus, mapping from word IDs to words (To be used in LdaModel's id2word parameter)

corpus, name_id_map = model.pre_process(newsgroup_data)

console.print("Finished PreProcessing!\n", style="success")
console.print("\tNumber of corpora = {}\n\tTotal number of characters = {}\n".format(len(newsgroup_data), char_count), style="info")


with console.status("Training the Model\n") as stat:
    trained_model, training_log = model.train(
        corpus=corpus, 
        num_topics=10, 
        id2word=name_id_map, 
        passes = 25, 
        random_state = 34)

console.print("training successful!", style="success")
console.print(training_log, style="info")

# topic_map = model.autoname_topics(trained_model)
topic_map = model.gen_topic_map(10)

# pprint(topic_map)
# for topic, prob in topics.items():
#     console.print("topic:", topic)
#     kw_p_pair = [item for item in prob.split("+")]
#     kw_p_pair_2 = [tuple(map(lambda x: x.strip(), (item.split('*')))) for item in kw_p_pair]
#     pprint(kw_p_pair_2)

console.print("Enter a string of text to identify it's topic!:", style="warning")
inp = console.input("\n")
print("\n")
# pprint(model.test_topic_distribution(inp, trained_model, topic_map))

for topic in model.test_topic_distribution(inp, trained_model, topic_map):
    console.print("[bold]Topic[/bold]: [green]{}[/green]\n[bold]Probability[/bold]: [green]{}[/green]".format(topic[0], topic[1]), style="info")