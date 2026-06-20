// ============================================================
//  BRAINCADE — Question Bank
//  Six lessons (Day 13–18) from Mr. Hsiao "Hue" Weng's AI & Coding course
//  Each question: text, four answers, index of correct answer (0-3),
//  and an explanation shown after answering.
// ============================================================

const LESSONS = [
  {
    id: "maze",
    day: 13,
    title: "Robot Maze & Search",
    emoji: "🤖",
    color: "#0b63ce",
    subtitle: "Grids, paths, and how a robot finds the goal",
    questions: [
      {
        q: "In the maze grid, what does the letter S stand for?",
        a: ["The goal square", "A wall", "The start square", "An open space"],
        correct: 2,
        why: "S marks the Start — the square where the robot begins its search."
      },
      {
        q: "What does the symbol # mean in the maze?",
        a: ["Open space", "A wall the robot can't pass", "The goal", "The start"],
        correct: 1,
        why: "# means a wall. The robot must check for walls before it moves."
      },
      {
        q: "Why does the robot keep a 'visited' set?",
        a: ["To remember where it has already been", "To count the walls", "To draw the maze", "To pick a random move"],
        correct: 0,
        why: "The visited set is the robot's memory so it doesn't repeat the same moves forever."
      },
      {
        q: "A 'stack' takes items out in what order?",
        a: ["The first item added comes out first", "A random item comes out", "The last item added comes out first", "The middle item comes out"],
        correct: 2,
        why: "A stack is last-in, first-out — like a stack of trays, you take the top one first."
      },
      {
        q: "What is an algorithm?",
        a: ["A type of wall", "A step-by-step rule for solving a problem", "A robot's name", "A random guess"],
        correct: 1,
        why: "An algorithm is a clear set of steps: check directions in order, move, remember, repeat."
      },
      {
        q: "If the robot had NO memory of visited squares, what could happen?",
        a: ["It would finish faster", "It might revisit squares again and again and get stuck", "It would never move", "It would find the shortest path"],
        correct: 1,
        why: "Without memory, the robot can loop over the same spaces forever, especially in a maze with loops."
      },
      {
        q: "What does the symbol G stand for in the maze?",
        a: ["The start", "A wall", "The goal the robot wants to reach", "An open path"],
        correct: 2,
        why: "G is the Goal — the search stops once the robot reaches it."
      },
      {
        q: "What does the symbol . (a dot) mean in the maze?",
        a: ["A wall", "Open space the robot can step on", "The goal", "The start"],
        correct: 1,
        why: "A dot is open space — a square the robot is allowed to move into."
      },
      {
        q: "Before moving into a square, what must the robot check?",
        a: ["Its color", "Whether it's a wall or off the grid", "How shiny it is", "The time of day"],
        correct: 1,
        why: "Boundary and wall checks keep the robot from leaving the grid or crashing into walls."
      },
      {
        q: "The robot checks directions in a fixed order (like up, right, down, left). Why?",
        a: ["To guess randomly", "So it follows the same rule every time — that's the algorithm", "To go faster", "To skip the goal"],
        correct: 1,
        why: "Following the same order each time makes the robot's behavior a predictable algorithm."
      },
      {
        q: "In grid[2][3], what do the numbers mean?",
        a: ["Row 2, column 3", "2 walls and 3 paths", "The score", "Column 2, the goal"],
        correct: 0,
        why: "A grid is rows and columns: grid[2][3] is the box at row 2, column 3."
      },
      {
        q: "If the robot changed its direction order, what would most likely happen?",
        a: ["It could never reach the goal", "It may still reach the goal, but draw a different path", "The maze disappears", "It stops working"],
        correct: 1,
        why: "A different order can still find the goal, but the path it takes can look very different."
      }
    ]
  },
  {
    id: "neuron",
    day: 14,
    title: "Tiny Neural Networks",
    emoji: "🧠",
    color: "#10b981",
    subtitle: "Weights, bias, error, and how a neuron learns",
    questions: [
      {
        q: "In a neuron, what does the 'weight' (w) control?",
        a: ["How much the neuron cares about an input", "The speed of the computer", "The color of the output", "How many inputs there are"],
        correct: 0,
        why: "A higher weight means that input matters more to the neuron's final answer."
      },
      {
        q: "How do we calculate the error?",
        a: ["weight times bias", "target minus prediction", "input plus output", "prediction times learning rate"],
        correct: 1,
        why: "error = target − prediction. Zero means a perfect guess."
      },
      {
        q: "What is the 'learning rate' (lr)?",
        a: ["The correct answer", "How big each update step is", "The number of inputs", "The final prediction"],
        correct: 1,
        why: "The learning rate is the step size. Too big overshoots; too small trains slowly."
      },
      {
        q: "What is the forward pass formula for one input?",
        a: ["raw = w − x − b", "raw = w + x + b", "raw = w × x + b", "raw = error × lr"],
        correct: 2,
        why: "raw = w × x + b. Then a threshold turns raw into a prediction of 0 or 1."
      },
      {
        q: "If the neuron predicts 0 but the target is 1, the error is...",
        a: ["Positive (+1)", "Negative (−1)", "Zero", "Always 10"],
        correct: 0,
        why: "error = target − pred = 1 − 0 = +1, so the weight increases to be more sensitive next time."
      },
      {
        q: "When the error is 0, what happens to the weight?",
        a: ["It doubles", "It stays the same — no update", "It resets to zero", "It becomes negative"],
        correct: 1,
        why: "w = w + lr × 0 × x = w. We were right, so the weights stay put."
      },
      {
        q: "What is the 'bias' (b) in a neuron?",
        a: ["The correct answer", "A nudge that shifts the output up or down, even when inputs are zero", "The number of inputs", "A type of error"],
        correct: 1,
        why: "Bias is like a head-start or handicap — it shifts the result even before inputs arrive."
      },
      {
        q: "What is the 'target' in training?",
        a: ["The neuron's guess", "The correct answer we want the neuron to produce", "The learning rate", "A random number"],
        correct: 1,
        why: "The target is the ground truth — the right answer (1 for 'big', 0 for 'small')."
      },
      {
        q: "What is the update rule for the weight?",
        a: ["w = w − x", "w = w + lr × error × x", "w = error × target", "w = b + x"],
        correct: 1,
        why: "w = w + lr × error × x nudges the weight in the direction that reduces the error."
      },
      {
        q: "In the loop 'for step in range(40)', how many training steps run?",
        a: ["4", "40", "400", "It runs forever"],
        correct: 1,
        why: "range(40) runs the training loop 40 times — 40 chances to learn."
      },
      {
        q: "Why does the neuron need RANDOM inputs during training?",
        a: ["To look pretty", "So it learns from many different examples, not just one number", "To slow it down", "It doesn't need them"],
        correct: 1,
        why: "If x were always the same, the neuron would only ever learn that one case."
      },
      {
        q: "A learning rate that is TOO large will most likely cause the neuron to...",
        a: ["Learn perfectly instantly", "Overshoot and bounce around instead of settling", "Stop using inputs", "Delete its weights"],
        correct: 1,
        why: "Too-big steps overshoot the right answer; too-small steps train very slowly."
      }
    ]
  },
  {
    id: "recommend",
    day: 15,
    title: "Recommendation Systems",
    emoji: "🍦",
    color: "#f59e0b",
    subtitle: "How apps pick 'what's next' using votes and averages",
    questions: [
      {
        q: "In our picker, a 'vote' is...",
        a: ["A list of games", "A thumbs-up (1) or thumbs-down (0) from one person", "The total score", "A type of loop"],
        correct: 1,
        why: "A vote is one person's 1 (thumbs-up) or 0 (thumbs-down) about one thing."
      },
      {
        q: "How do you find the average of votes [1, 1, 0, 1, 1]?",
        a: ["Add them: 4", "Multiply: 0", "Add then divide by 5: 0.8", "Pick the biggest: 1"],
        correct: 2,
        why: "Add the votes (4), divide by how many voters (5): 4 ÷ 5 = 0.8, an 80% score."
      },
      {
        q: "What does the picker recommend first?",
        a: ["The newest item", "A random item", "The item with the highest average ('Top')", "The item with the fewest votes"],
        correct: 2,
        why: "The 'Top' item has the highest average, so the picker shows it first."
      },
      {
        q: "In Python, max(averages, key=averages.get) finds...",
        a: ["The smallest score", "The biggest score by its average value", "The first game", "The number of games"],
        correct: 1,
        why: "max finds the biggest, and key=averages.get tells it to compare by score."
      },
      {
        q: "Why can 'most popular' sometimes pick the wrong thing for YOU?",
        a: ["The math is broken", "Your taste may differ from the group", "Averages are always wrong", "Computers can't add"],
        correct: 1,
        why: "If 19 kids love chocolate but you love mint, the popular pick isn't your pick."
      },
      {
        q: "Giving each person their own ratings list is called...",
        a: ["A wall", "Personalization", "A stack", "An edge"],
        correct: 1,
        why: "Personalization runs the same average rule once per person — the secret behind apps like YouTube."
      },
      {
        q: "In Python, what is the average of the votes [1, 0, 0, 1, 0]?",
        a: ["0.4", "0.8", "2.0", "0.2"],
        correct: 0,
        why: "1+0+0+1+0 = 2, and 2 ÷ 5 = 0.4."
      },
      {
        q: "In our recommender, the name of each game is the ___ and its votes are the ___.",
        a: ["value / key", "key / value", "average / top", "loop / list"],
        correct: 1,
        why: "In a dictionary, the game name is the key and its list of votes is the value."
      },
      {
        q: "A dictionary in Python is best described as...",
        a: ["A single number", "A labeled box that links keys to values", "A type of loop", "A wall"],
        correct: 1,
        why: "A dictionary stores key→value pairs, like labels on boxes of stuff."
      },
      {
        q: "What does sum(votes) / len(votes) calculate?",
        a: ["The biggest vote", "The average of the votes", "The number of games", "A random pick"],
        correct: 1,
        why: "sum adds the votes and len counts them — together they give the average."
      },
      {
        q: "Why might a brand-new item never get recommended by a 'most popular' picker?",
        a: ["It has zero votes, so its average can't be on top yet", "It's too expensive", "The code is broken", "It has too many votes"],
        correct: 0,
        why: "With no votes yet, a new item can't reach the top — even if it's actually great."
      },
      {
        q: "What is the term for an app showing you the same kind of thing over and over?",
        a: ["A speed bonus", "A feedback loop / filter bubble", "An average", "A tuple"],
        correct: 1,
        why: "Only seeing popular picks can trap you in a loop where you never try anything new."
      }
    ]
  },
  {
    id: "images",
    day: 16,
    title: "AI for Images",
    emoji: "🖼️",
    color: "#06b6d4",
    subtitle: "Pixels, edges, and how computers learn to see",
    questions: [
      {
        q: "What is a pixel?",
        a: ["The whole photo", "The tiniest dot in a picture, storing one number", "A type of loop", "The camera"],
        correct: 1,
        why: "A pixel is the smallest square in a picture — like one box on graph paper — and stores one number."
      },
      {
        q: "In our mini-grid, what does the number 0 represent?",
        a: ["Very bright", "Completely black", "An edge", "The center"],
        correct: 1,
        why: "0 = completely black (dark), and 9 = very bright in our 5×5 mini-picture."
      },
      {
        q: "An edge in an image is where...",
        a: ["The numbers stay the same", "The numbers suddenly change a lot", "The photo ends", "The colors blur together"],
        correct: 1,
        why: "An edge is where pixel numbers jump quickly — like dark sky meeting a bright building."
      },
      {
        q: "What is the edge formula for a pixel?",
        a: ["value = center + neighbors", "value = center × 4 − (up + down + left + right)", "value = center ÷ 4", "value = up × down"],
        correct: 1,
        why: "We compare the center to its 4 neighbors: a big result means the pixel is on an edge."
      },
      {
        q: "For the center pixel 9 with neighbors 5,5,5,5, what is the value?",
        a: ["0", "16", "20", "10"],
        correct: 1,
        why: "9 × 4 = 36, and 5+5+5+5 = 20, so 36 − 20 = 16."
      },
      {
        q: "Why does AI need to find edges?",
        a: ["To make photos bigger", "Edges reveal shapes, which help AI recognize objects", "To delete pixels", "To slow the computer down"],
        correct: 1,
        why: "If AI can find edges, it can see shapes that tell it 'this is a face' or 'this is a stop sign'."
      },
      {
        q: "In our edge code, a 'nested loop' means...",
        a: ["One loop only", "A loop inside another loop — rows outside, columns inside", "No loops", "A loop that never ends"],
        correct: 1,
        why: "The outer loop moves through rows and the inner loop through columns, visiting every pixel."
      },
      {
        q: "What is a 'filter' in image processing?",
        a: ["A photo album", "A math operation that slides over pixels and checks neighbors", "A color", "A pixel"],
        correct: 1,
        why: "A filter slides over the image; different filters can blur, sharpen, or find edges."
      },
      {
        q: "Why does the edge code skip row 0 (start at row 1)?",
        a: ["Row 0 is the goal", "There's no row above row 0, so 'up' would go off the grid", "Row 0 is a wall", "To save time"],
        correct: 1,
        why: "Reading image[y-1] for row 0 would fall off the top of the grid, so we skip the border."
      },
      {
        q: "After edge detection, a BIG result value means the pixel is...",
        a: ["In a flat area", "On an edge", "Black", "Deleted"],
        correct: 1,
        why: "A big difference from its neighbors means the pixel sits on an edge."
      },
      {
        q: "How many neighbors does our edge formula compare each pixel to?",
        a: ["2", "4 (up, down, left, right)", "8", "1"],
        correct: 1,
        why: "value = center×4 − (up + down + left + right) uses the 4 direct neighbors."
      },
      {
        q: "For a pixel surrounded by neighbors that all equal its own value, the edge result is...",
        a: ["A huge number", "Zero (a flat area)", "Negative 100", "Always 9"],
        correct: 1,
        why: "center×4 − (four equal neighbors) = 0, meaning no edge — it's a flat region."
      }
    ]
  },
  {
    id: "chatbot",
    day: 17,
    title: "Chatbots & Language AI",
    emoji: "💬",
    color: "#8b5cf6",
    subtitle: "Rule-based bots, intent, and where they fail",
    questions: [
      {
        q: "What is a 'rule-based' chatbot?",
        a: ["A bot that learns on its own", "A bot where every reply is written by a human developer", "A bot with no replies", "A bot that uses billions of weights"],
        correct: 1,
        why: "In a rule-based bot, humans write every reply. No learning happens — only pattern-matching."
      },
      {
        q: "In AI, 'intent' means...",
        a: ["The color of the text", "The goal behind a message", "The length of a message", "The bot's name"],
        correct: 1,
        why: "Intent is the purpose of a message: 'Hey!' = greeting intent, 'I'm sad' = negative-feeling intent."
      },
      {
        q: "Why does TinyBot call .lower().strip() on a message?",
        a: ["To make the message longer", "To make text lowercase and remove spaces so matching works", "To delete the message", "To translate it"],
        correct: 1,
        why: "It turns 'HELLO' into 'hello' and trims spaces so keyword matching is reliable."
      },
      {
        q: "TinyBot replies to 'I am not sad' as if you ARE sad. This is a...",
        a: ["Correct reply", "False positive — it found 'sad' but ignored 'not'", "Synonym match", "Safety feature"],
        correct: 1,
        why: "Keyword matching is brittle: it spots 'sad' but misses the word 'not' that flips the meaning."
      },
      {
        q: "What does 'keyword match' miss most easily?",
        a: ["Numbers", "Synonyms and context (like 'furious' for angry)", "Letters", "Spaces"],
        correct: 1,
        why: "'I'm furious' falls to else because the rules only know 'angry', not its synonym."
      },
      {
        q: "An AI chatbot like ChatGPT uses... instead of hand-written rules.",
        a: ["A single if-statement", "Billions of learned numerical weights", "One keyword", "A maze"],
        correct: 1,
        why: "ChatGPT-style bots learn billions of weights from text, rather than relying on a few human rules."
      },
      {
        q: "What does NLP stand for?",
        a: ["New Language Print", "Natural Language Processing", "Neural Loop Program", "Next Letter Pick"],
        correct: 1,
        why: "NLP is the AI field that teaches computers to read, understand, and respond to human language."
      },
      {
        q: "What is the FIRST step of the chatbot loop?",
        a: ["Respond", "Receive the user's message", "Match intent", "Delete the message"],
        correct: 1,
        why: "Step 1 is Receive: the user types a message and Python stores it as a string."
      },
      {
        q: "In TinyBot, what does 'while True' do?",
        a: ["Runs once", "Loops forever until the user types 'bye'", "Skips the chat", "Counts to ten"],
        correct: 1,
        why: "while True keeps the conversation going until a break (typing 'bye') ends it."
      },
      {
        q: "Why can the word 'hi' inside 'this' confuse a keyword-matching bot?",
        a: ["It can't — bots are perfect", "'hi' appears inside 't-hi-s', so a loose check may falsely match a greeting", "'this' is a greeting", "It deletes the message"],
        correct: 1,
        why: "Checking 'hi' in msg can wrongly trigger on words that merely contain those letters."
      },
      {
        q: "What is a key SAFETY risk of TinyBot's 'else' branch?",
        a: ["It greets too much", "It might give an accidental reply to a harmful question", "It uses too many emojis", "It runs too fast"],
        correct: 1,
        why: "The catch-all else can accidentally respond to questions a school bot should never answer."
      },
      {
        q: "TinyBot treats every message fresh and forgets the last one. This means it has no...",
        a: ["Keywords", "Context memory of the conversation", "Replies", "Loop"],
        correct: 1,
        why: "Without context memory, TinyBot can't follow the flow of a conversation."
      }
    ]
  },
  {
    id: "bias",
    day: 18,
    title: "Data, Bias & Fairness",
    emoji: "⚖️",
    color: "#ef4444",
    subtitle: "How data choices create biased AI — and how to fix it",
    questions: [
      {
        q: "What is 'training data'?",
        a: ["The robot's body", "Examples the model learns its patterns from", "A type of wall", "The final answer"],
        correct: 1,
        why: "Training data is the examples a model learns from — its quality decides the output quality."
      },
      {
        q: "'Class imbalance' means...",
        a: ["All categories are equal", "One category has far more examples than another", "The data is perfect", "There are no labels"],
        correct: 1,
        why: "Example: 900 cat images vs 10 dog images. The model gets great at cats, poor at dogs."
      },
      {
        q: "In Python, what does Counter(['a','b','a']) produce?",
        a: ["['a','b','a']", "{'a':2,'b':1}", "3", "['a','b']"],
        correct: 1,
        why: "Counter automatically tallies items: 'a' appears twice, 'b' once."
      },
      {
        q: "A tuple like ('Alex', 'red') is best described as...",
        a: ["A changeable list", "An immutable pair, ideal for a record", "A loop", "A dictionary"],
        correct: 1,
        why: "A tuple is an immutable (unchangeable) pair — perfect for storing a fixed record."
      },
      {
        q: "Amazon's hiring AI became biased because it was trained on resumes that were...",
        a: ["96% male", "All from women", "Perfectly balanced", "Made up"],
        correct: 0,
        why: "Trained on 10 years of 96%-male resumes, it learned to penalize CVs containing 'women's'."
      },
      {
        q: "Could a dataset of only TRUE facts still create a biased AI?",
        a: ["No, true facts are always fair", "Yes — if the data over-represents one group, the model skews", "Only if the facts are wrong", "Never"],
        correct: 1,
        why: "Even all-true data can be biased if it over-represents one group, like one city or one ethnicity."
      },
      {
        q: "In Python, what does Counter('banana').most_common(2) return?",
        a: ["[('a', 3), ('n', 2)]", "[('b', 1)]", "['banana']", "6"],
        correct: 0,
        why: "'a' appears 3 times and 'n' 2 times, so the top two are ('a',3) and ('n',2)."
      },
      {
        q: "A 'fairness metric' is...",
        a: ["The model's speed", "A number that measures how equally a model treats different groups", "The training data size", "A type of tuple"],
        correct: 1,
        why: "Fairness metrics put a number on how evenly a model performs across groups."
      },
      {
        q: "In the MIT face-recognition study, error was lowest for light-skinned men and highest for...",
        a: ["Light-skinned women", "Dark-skinned women (up to ~35%)", "Robots", "Everyone equally"],
        correct: 1,
        why: "Errors ranged from about 0.8% for light-skinned men to ~34.7% for dark-skinned women."
      },
      {
        q: "A list comprehension like [x*2 for x in nums if x>0] does what in one line?",
        a: ["Nothing", "Filters and transforms a list", "Deletes the list", "Counts letters"],
        correct: 1,
        why: "It filters (if x>0) and transforms (x*2) the items into a new list in a single line."
      },
      {
        q: "A robot trained only on data from ONE city may fail elsewhere because...",
        a: ["Cities are all identical", "Its training data doesn't represent other places", "It runs out of memory", "It has no weights"],
        correct: 1,
        why: "If the data only reflects one place, the model's 'normal' won't match different cities."
      },
      {
        q: "A model that gets cats right 99% but dogs only 20% (trained on 1000 cats, 10 dogs) shows...",
        a: ["Great, balanced performance", "Bias from class imbalance", "A perfect model", "No problem at all"],
        correct: 1,
        why: "With far more cat examples, the model learned cats well but is unfair to the rare dog class."
      }
    ]
  }
];

// Build a "Mixed / All Lessons" deck by sampling across lessons.
function buildMixedDeck() {
  const all = [];
  LESSONS.forEach(l => l.questions.forEach(q => all.push({ ...q, lessonId: l.id, lessonTitle: l.title, lessonColor: l.color, emoji: l.emoji })));
  return all;
}
