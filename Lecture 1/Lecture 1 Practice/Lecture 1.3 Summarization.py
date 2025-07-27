#Lecture 1 Prompt Engineering with Jupyter Notebook

import openai

# Initialize the OpenAI client
client = openai.OpenAI(api_key='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')

def get_completion(prompt, model="gpt-4o-mini"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

text = """
In an office at MIT’s Computer Science and Artificial Intelligence Laboratory (CSAIL), a soft robotic hand carefully curls its fingers to grasp a small object. The intriguing part isn’t the mechanical design or embedded sensors — in fact, the hand contains none. Instead, the entire system relies on a single camera that watches the robot’s movements and uses that visual data to control it.

This capability comes from a new system CSAIL scientists developed, offering a different perspective on robotic control. Rather than using hand-designed models or complex sensor arrays, it allows robots to learn how their bodies respond to control commands, solely through vision. The approach, called Neural Jacobian Fields (NJF), gives robots a kind of bodily self-awareness. An open-access paper about the work was published in Nature on June 25.

“This work points to a shift from programming robots to teaching robots,” says Sizhe Lester Li, MIT PhD student in electrical engineering and computer science, CSAIL affiliate, and lead researcher on the work. “Today, many robotics tasks require extensive engineering and coding. In the future, we envision showing a robot what to do, and letting it learn how to achieve the goal autonomously.”

The motivation stems from a simple but powerful reframing: The main barrier to affordable, flexible robotics isn't hardware — it’s control of capability, which could be achieved in multiple ways. Traditional robots are built to be rigid and sensor-rich, making it easier to construct a digital twin, a precise mathematical replica used for control. But when a robot is soft, deformable, or irregularly shaped, those assumptions fall apart. Rather than forcing robots to match our models, NJF flips the script — giving robots the ability to learn their own internal model from observation.

Look and learn

This decoupling of modeling and hardware design could significantly expand the design space for robotics. In soft and bio-inspired robots, designers often embed sensors or reinforce parts of the structure just to make modeling feasible. NJF lifts that constraint. The system doesn’t need onboard sensors or design tweaks to make control possible. Designers are freer to explore unconventional, unconstrained morphologies without worrying about whether they’ll be able to model or control them later.

“Think about how you learn to control your fingers: you wiggle, you observe, you adapt,” says Li. “That’s what our system does. It experiments with random actions and figures out which controls move which parts of the robot.”

The system has proven robust across a range of robot types. The team tested NJF on a pneumatic soft robotic hand capable of pinching and grasping, a rigid Allegro hand, a 3D-printed robotic arm, and even a rotating platform with no embedded sensors. In every case, the system learned both the robot’s shape and how it responded to control signals, just from vision and random motion.

The researchers see potential far beyond the lab. Robots equipped with NJF could one day perform agricultural tasks with centimeter-level localization accuracy, operate on construction sites without elaborate sensor arrays, or navigate dynamic environments where traditional methods break down.

At the core of NJF is a neural network that captures two intertwined aspects of a robot’s embodiment: its three-dimensional geometry and its sensitivity to control inputs. The system builds on neural radiance fields (NeRF), a technique that reconstructs 3D scenes from images by mapping spatial coordinates to color and density values. NJF extends this approach by learning not only the robot’s shape, but also a Jacobian field, a function that predicts how any point on the robot’s body moves in response to motor commands.

To train the model, the robot performs random motions while multiple cameras record the outcomes. No human supervision or prior knowledge of the robot’s structure is required — the system simply infers the relationship between control signals and motion by watching.

Once training is complete, the robot only needs a single monocular camera for real-time closed-loop control, running at about 12 Hertz. This allows it to continuously observe itself, plan, and act responsively. That speed makes NJF more viable than many physics-based simulators for soft robots, which are often too computationally intensive for real-time use.

In early simulations, even simple 2D fingers and sliders were able to learn this mapping using just a few examples. By modeling how specific points deform or shift in response to action, NJF builds a dense map of controllability. That internal model allows it to generalize motion across the robot’s body, even when the data are noisy or incomplete.

“What’s really interesting is that the system figures out on its own which motors control which parts of the robot,” says Li. “This isn’t programmed — it emerges naturally through learning, much like a person discovering the buttons on a new device.”

The future is soft

For decades, robotics has favored rigid, easily modeled machines — like the industrial arms found in factories — because their properties simplify control. But the field has been moving toward soft, bio-inspired robots that can adapt to the real world more fluidly. The trade-off? These robots are harder to model.

“Robotics today often feels out of reach because of costly sensors and complex programming. Our goal with Neural Jacobian Fields is to lower the barrier, making robotics affordable, adaptable, and accessible to more people. Vision is a resilient, reliable sensor,” says senior author and MIT Assistant Professor Vincent Sitzmann, who leads the Scene Representation group. “It opens the door to robots that can operate in messy, unstructured environments, from farms to construction sites, without expensive infrastructure.”

“Vision alone can provide the cues needed for localization and control — eliminating the need for GPS, external tracking systems, or complex onboard sensors. This opens the door to robust, adaptive behavior in unstructured environments, from drones navigating indoors or underground without maps to mobile manipulators working in cluttered homes or warehouses, and even legged robots traversing uneven terrain,” says co-author Daniela Rus, MIT professor of electrical engineering and computer science and director of CSAIL. “By learning from visual feedback, these systems develop internal models of their own motion and dynamics, enabling flexible, self-supervised operation where traditional localization methods would fail.”

While training NJF currently requires multiple cameras and must be redone for each robot, the researchers are already imagining a more accessible version. In the future, hobbyists could record a robot’s random movements with their phone, much like you’d take a video of a rental car before driving off, and use that footage to create a control model, with no prior knowledge or special equipment required.

The system doesn’t yet generalize across different robots, and it lacks force or tactile sensing, limiting its effectiveness on contact-rich tasks. But the team is exploring new ways to address these limitations: improving generalization, handling occlusions, and extending the model’s ability to reason over longer spatial and temporal horizons.

“Just as humans develop an intuitive understanding of how their bodies move and respond to commands, NJF gives robots that kind of embodied self-awareness through vision alone,” says Li. “This understanding is a foundation for flexible manipulation and control in real-world environments. Our work, essentially, reflects a broader trend in robotics: moving away from manually programming detailed models toward teaching robots through observation and interaction.”

This paper brought together the computer vision and self-supervised learning work from the Sitzmann lab and the expertise in soft robots from the Rus lab. Li, Sitzmann, and Rus co-authored the paper with CSAIL affiliates Annan Zhang SM ’22, a PhD student in electrical engineering and computer science (EECS); Boyuan Chen, a PhD student in EECS; Hanna Matusik, an undergraduate researcher in mechanical engineering; and Chao Liu, a postdoc in the Senseable City Lab at MIT. 

The research was supported by the Solomon Buchsbaum Research Fund through MIT’s Research Support Committee, an MIT Presidential Fellowship, the National Science Foundation, and the Gwangju Institute of Science and Technology.
"""

prompt = f"Summarize the following text with one paragraph:\n{text}"
response = get_completion(prompt)
print(response)
