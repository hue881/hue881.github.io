// ============================================================
//  BRAINCADE — Question Bank
//  Modern Chinese School * Summer 2026 AI & Coding Course
//  Six thematic decks spanning the full course arc: computational
//  thinking, Scratch, Python basics, Python logic & games, applied AI,
//  and the capstone/shipping process.
//  Each question: text, four answers, index of correct answer (0-3),
//  and an explanation shown after answering.
// ============================================================

const LESSONS = [
  {
    id: "maze",
    day: 1,
    title: "Computational Thinking & Algorithms",
    emoji: "\ud83e\udde0",
    color: "#0b63ce",
    subtitle: "Algorithms, internet safety, and how AI can be wrong",
    questions: [
      {
        q: "What is an algorithm?",
        a: ["A random guess", "A precise, ordered list of steps that solves a task", "A type of computer virus", "A robot's name"],
        correct: 1,
        why: "An algorithm is a precise list of steps, in order, that solves a task \u2014 like a recipe for food."
      },
      {
        q: "Why does a computer \"freeze\" if you skip a step when giving it instructions (like putting on a shoe)?",
        a: ["It needs every step spelled out — it can't fill in gaps like a brain does", "Computers don't like shoes", "It only understands numbers", "Computers are broken"],
        correct: 0,
        why: "A human brain automatically fills in hundreds of tiny steps. A computer needs every detail said explicitly."
      },
      {
        q: "What is one single step in a list of instructions called?",
        a: ["An instruction", "A bug", "A variable", "An algorithm"],
        correct: 0,
        why: "An instruction is one single step. Computers do each instruction exactly and in order."
      },
      {
        q: "What is a bug?",
        a: ["A private password", "A type of algorithm", "A helpful shortcut", "A mistake in your steps that makes things go wrong"],
        correct: 3,
        why: "A bug is a mistake in your steps. Bugs are normal \u2014 programmers fix them all the time."
      },
      {
        q: "According to the 'Are You Smarter Than a Computer?' lesson, what is the big lesson about AI?",
        a: ["AI never makes mistakes", "AI can read your mind", "AI is still just following instructions and patterns — the smarter your instructions, the smarter the result", "AI doesn't need any instructions"],
        correct: 2,
        why: "AI is still just following instructions and patterns. It can't read your mind \u2014 clearer instructions lead to better results."
      },
      {
        q: "What should you NEVER share online, according to the internet safety lesson?",
        a: ["What game you like", "Your favorite animal", "Your favorite color", "Your full name, address, school, or passwords"],
        correct: 3,
        why: "Private information like your full name, address, school, and passwords is like a key to your house \u2014 never share it online."
      },
      {
        q: "What is one way AI can go wrong, according to the 'Good Robot / Bad Robot' lesson?",
        a: ["If training data has mistakes, AI repeats them, and it can confidently make up facts that sound real but aren't", "It can only add numbers", "It never makes decisions", "It always tells the truth"],
        correct: 0,
        why: "AI repeats mistakes found in its training data, and it can state made-up 'facts' confidently. AI is a helpful tool, not the final answer."
      },
      {
        q: "If something online makes you feel uncomfortable or confused, what should you do?",
        a: ["Reply immediately", "Ignore it and keep browsing", "Share it with more people", "Tell a parent, teacher, or trusted adult"],
        correct: 3,
        why: "You should always tell a trusted adult. You will never get in trouble for speaking up about something that felt wrong online."
      },
      {
        q: "A decision tree that sorts a photo as 'cat' or 'not cat' by checking traits like whiskers and 'says meow' is an example of what?",
        a: ["A classifier", "A virus", "A random number generator", "A password"],
        correct: 0,
        why: "This is a classifier \u2014 an AI system that sorts inputs into categories using rules, just like the class's hand-drawn decision tree."
      },
      {
        q: "Why might a hand-built 'Is it a cat?' decision tree get confused by a lion?",
        a: ["The tree's rules were written for typical house cats and don't cover every case, showing a limit of hand-written rules", "The tree would always say yes", "Lions can't be photographed", "Lions don't have whiskers"],
        correct: 0,
        why: "Hand-written rules only cover the cases the rule-writer thought of. This is why modern AI instead learns rules from thousands of examples."
      },
      {
        q: "How is modern AI different from early rule-based systems like a hand-built decision tree?",
        a: ["There is no difference", "Modern AI has no rules at all", "Modern AI learns the rules from thousands of examples instead of a human writing every rule by hand", "Modern AI is written in Scratch"],
        correct: 2,
        why: "Early AI was built from hand-written if-then rules. Modern AI learns patterns from data \u2014 the 'rules' become numbers called weights."
      }
    ,
      {
        q: "What do we call the starting point in a maze-solving algorithm?",
        a: ["The wall", "The start node", "The loop", "The goal"],
        correct: 1,
        why: "The start node is where the search begins before exploring paths toward the goal."
      },
      {
        q: "What do we call the destination in a maze-solving algorithm?",
        a: ["The variable", "The start node", "The goal or end node", "The obstacle"],
        correct: 2,
        why: "The goal (or end) node is the destination the algorithm is trying to reach."
      },
      {
        q: "Why do algorithms need to be precise instead of vague?",
        a: ["A computer can't guess what you mean — every step must be exact and unambiguous", "Computers understand vague English perfectly", "Precision only matters for math", "Vague steps are always faster"],
        correct: 0,
        why: "Computers cannot infer intent. Every step must be spelled out exactly so there is no room for misinterpretation."
      },
      {
        q: "What is a 'wall' in a maze-solving algorithm's grid?",
        a: ["A blocked cell the algorithm must avoid or route around", "The starting cell", "A path the algorithm can freely walk through", "The variable holding the score"],
        correct: 0,
        why: "Walls represent blocked cells. A correct maze algorithm must never route a path through them."
      },
      {
        q: "Why is testing an algorithm on multiple different mazes important?",
        a: ["One maze might accidentally work by luck; testing many mazes reveals real bugs", "Testing only wastes time", "Mazes never have bugs", "It isn't important, one test is enough"],
        correct: 0,
        why: "An algorithm might seem to work on one lucky example. Testing several different cases uncovers hidden bugs."
      },
      {
        q: "What does 'debugging' an algorithm mean?",
        a: ["Finding and fixing the mistakes in your steps", "Renaming your variables", "Deleting the whole algorithm", "Adding more bugs on purpose"],
        correct: 0,
        why: "Debugging means locating the mistake in your logic or steps and correcting it so the algorithm works as intended."
      },
      {
        q: "Why should you never share your home address with strangers online?",
        a: ["Addresses are public information anyway", "Personal information can be used to find or harm you in real life", "It's a fun way to make friends", "It has no real risk"],
        correct: 1,
        why: "Personal details like your address can let strangers locate you in real life, which is why they must stay private."
      },
      {
        q: "What is an example of a 'private' piece of information you should protect online?",
        a: ["Your school name and daily schedule", "A joke you like", "Your favorite movie", "The weather today"],
        correct: 0,
        why: "Details like your school and schedule can reveal where you'll be and when, so they should never be shared publicly."
      },
      {
        q: "Why can AI 'confidently state something false'?",
        a: ["AI always checks facts before answering", "AI refuses to answer if unsure", "AI predicts likely-sounding patterns from its training data, which can produce wrong but convincing text", "AI has human fact-checkers reviewing every answer"],
        correct: 2,
        why: "AI generates text based on learned patterns, not verified truth, so it can produce false statements that sound confident and real."
      },
      {
        q: "What is one reason a decision tree classifier might mislabel a 'cat' picture as 'not a cat'?",
        a: ["Cats cannot be photographed", "The tree has no rules at all", "The classifier only accepts video, not photos", "The picture doesn't match the specific traits the tree was built to check"],
        correct: 3,
        why: "A hand-built decision tree only checks the traits its rules cover, so unusual cat photos that don't match those traits get misclassified."
      },
      {
        q: "In an algorithm, why does the ORDER of steps matter?",
        a: ["Only the last step matters", "Doing steps out of order can produce a completely different, often wrong, result", "Computers automatically reorder steps for you", "Order never matters, any order works"],
        correct: 1,
        why: "Algorithms execute steps in sequence; changing the order can change what happens, just like putting on socks after shoes."
      },
      {
        q: "What is the purpose of 'internet safety' lessons for young programmers?",
        a: ["To replace algorithm lessons", "To teach a new programming language", "To teach how to recognize risks and protect personal information while using technology", "To scare students away from computers"],
        correct: 2,
        why: "Internet safety lessons teach how to recognize risks online and protect personal information, an essential skill for anyone using technology."
      },
      {
        q: "Why might two different algorithms both solve the same maze correctly but at different speeds?",
        a: ["Only the color of the code affects speed", "Some approaches explore fewer unnecessary paths, making them more efficient", "Speed differences are impossible", "All algorithms are always equally fast"],
        correct: 1,
        why: "Different algorithms can reach the same correct answer via different amounts of exploration, so efficiency can vary even when both are correct."
      },
      {
        q: "What should you check first if your maze-solving algorithm never reaches the goal?",
        a: ["Change the color scheme", "Add more walls", "Whether the start, goal, and wall steps were defined and followed correctly", "Immediately give up"],
        correct: 2,
        why: "If a maze algorithm fails to reach the goal, the first step is checking whether the start, goal, and movement steps are defined and followed correctly."
      }
    ]
  },
  {
    id: "neuron",
    day: 2,
    title: "Scratch: Building Blocks",
    emoji: "\ud83d\udc31",
    color: "#10b981",
    subtitle: "Sequences, events, loops, conditionals & variables",
    questions: [
      {
        q: "What is a sequence in Scratch?",
        a: ["A sound effect", "A list of steps carried out in order, one at a time", "A type of sprite", "A random block"],
        correct: 1,
        why: "A sequence is steps executed one after another, in order \u2014 like a recipe. Skip a step and the result changes."
      },
      {
        q: "What is an event in Scratch?",
        a: ["A background image", "A math operation", "A costume change", "A trigger that starts a script running"],
        correct: 3,
        why: "An event is a trigger \u2014 'when ___ happens, do ___.' Without an event, a script never runs."
      },
      {
        q: "What happens to a Scratch block that is NOT connected to a hat block (like 'when green flag clicked')?",
        a: ["It never runs", "It deletes itself", "It runs first", "It runs twice"],
        correct: 0,
        why: "Disconnected blocks never run. Every script needs a hat/event block to trigger it."
      },
      {
        q: "Why use a 'repeat' loop instead of copying the same blocks 10 times?",
        a: ["It's shorter, and to change the count you only edit one number", "There's no difference", "Repeat blocks use more energy", "Repeat blocks run slower"],
        correct: 0,
        why: "A loop turns 10 repeated lines into just a few \u2014 and changing the loop count is a one-number edit instead of rewriting everything."
      },
      {
        q: "When should you use a 'forever' loop instead of 'repeat N'?",
        a: ["When you want an exact, fixed number of repeats", "When the action should keep running until the program stops (like a bouncing ball)", "Never — forever loops are broken", "Only for sound effects"],
        correct: 1,
        why: "'Repeat N' is for a fixed count. 'Forever' is for actions that should continue until the user stops the program."
      },
      {
        q: "In Scratch, what shape of block fits into the hexagon slot of an 'if' block?",
        a: ["A round (reporter) block", "Only sound blocks", "A hexagon (boolean/true-false) block", "Any block at all"],
        correct: 2,
        why: "The hexagon slot only accepts boolean (true/false) values \u2014 a visual hint that only true/false blocks fit there."
      },
      {
        q: "What is the difference between 'if' and 'if-else'?",
        a: ["'if' runs twice as fast", "They are identical", "'if' only runs code when true; 'if-else' always runs something, for both true and false", "'if-else' only works with numbers"],
        correct: 2,
        why: "The 'if' block only acts when the condition is TRUE. 'if-else' handles both cases \u2014 something happens whether the condition is true or false."
      },
      {
        q: "What is a variable in Scratch?",
        a: ["A background loop", "A sound file", "A type of costume", "A labeled container that stores one value you can read, change, or reset"],
        correct: 3,
        why: "A variable is a labeled box that stores a value \u2014 you can read it, change it, or reset it any time while the program runs."
      },
      {
        q: "What is the difference between 'set score to 0' and 'change score by 1'?",
        a: ["'set' can only be used once", "'set' replaces the value; 'change' adds to the current value", "'change' only works on text", "They do the same thing"],
        correct: 1,
        why: "'Set' replaces the value entirely. 'Change by' adds to whatever the current value already is."
      },
      {
        q: "Why should a script include 'set score to 0' under 'when green flag clicked'?",
        a: ["It's not necessary", "It makes the game run faster", "Otherwise the score starts wherever it ended in the last game instead of at zero", "It hides the score"],
        correct: 2,
        why: "Without resetting, the score would carry over from the last time the game was played instead of starting fresh."
      },
      {
        q: "How is a Scratch variable similar to how AI works?",
        a: ["An AI's 'confidence score' is stored as a variable, just like a game score", "They are unrelated", "Variables can only store images", "AI never uses variables"],
        correct: 0,
        why: "An AI model's confidence score is a variable \u2014 a container that stores how sure the model is, just like a game score variable."
      },
      {
        q: "A sprite can have multiple scripts. What does this allow?",
        a: ["Only the last script written will work", "Each script waits for its own event and they run independently without interfering", "Scripts must always run in the same order", "Only one script may ever run"],
        correct: 1,
        why: "Multiple scripts on one sprite each respond to their own event and run independently \u2014 they don't interfere with each other."
      }
    ,
      {
        q: "What visual shape do Scratch's 'hat' blocks have?",
        a: ["Square", "Rounded top, like a hat", "Triangle", "Hexagon"],
        correct: 1,
        why: "Hat blocks have a rounded top shape, resembling a hat, and sit at the top of a script to trigger it."
      },
      {
        q: "In Scratch, what shape are boolean (true/false) blocks?",
        a: ["Star", "Hexagon", "Square", "Circle"],
        correct: 1,
        why: "Boolean blocks are hexagon-shaped, which is why they only fit into the hexagon-shaped slots of if-blocks."
      },
      {
        q: "What happens when you click the green flag in Scratch?",
        a: ["All scripts with a 'when green flag clicked' hat block start running", "Nothing happens", "Only one sprite moves", "The project closes"],
        correct: 0,
        why: "Clicking the green flag triggers every script that begins with the 'when green flag clicked' event block."
      },
      {
        q: "What is the purpose of a 'wait' block in Scratch?",
        a: ["To permanently stop the script", "To increase the score", "To pause the script for a set amount of time before continuing", "To delete a variable"],
        correct: 2,
        why: "A wait block pauses the script for a specified duration before the next block runs, useful for timing animations or effects."
      },
      {
        q: "Why might you use a 'repeat until' block instead of a plain 'repeat 10' block?",
        a: ["'Repeat until' cannot use conditions", "They are exactly identical", "'Repeat until' keeps looping until a condition becomes true, useful when you don't know the exact count in advance", "'Repeat until' runs a fixed number of times only"],
        correct: 2,
        why: "'Repeat until' loops based on a condition rather than a fixed count, which is ideal when the number of repeats isn't known ahead of time."
      },
      {
        q: "What does broadcasting a message in Scratch allow sprites to do?",
        a: ["Delete each other", "Communicate and trigger actions in other sprites that are listening for that message", "Change color randomly", "Nothing, broadcasts have no effect"],
        correct: 1,
        why: "Broadcast and 'when I receive' blocks let sprites communicate, so one sprite's action can trigger a response in another."
      },
      {
        q: "In Scratch, what does the 'change x by 10' block do to a sprite?",
        a: ["Moves the sprite 10 steps to the right on the x-axis", "Deletes the sprite", "Rotates the sprite 10 degrees", "Changes the sprite's costume"],
        correct: 0,
        why: "'Change x by 10' shifts the sprite's horizontal position, moving it right if positive or left if negative."
      },
      {
        q: "What is the role of a 'costume' in Scratch?",
        a: ["A costume is a sound effect", "A costume is a type of variable", "A costume stores a number", "A costume is a visual appearance a sprite can switch between, useful for animation"],
        correct: 3,
        why: "Costumes are different visual appearances of a sprite; switching between them creates animation effects like walking or blinking."
      },
      {
        q: "Why is it useful to name your variables clearly, like 'score' instead of 'x1'?",
        a: ["Clear names make code easier to understand and debug, especially in bigger projects", "Clear names make the program run faster", "Scratch requires single-letter names", "Names don't matter at all to Scratch"],
        correct: 0,
        why: "Clear variable names make code much easier to read, understand, and debug, especially as projects grow larger."
      },
      {
        q: "What does the 'if touching' block check for in Scratch?",
        a: ["The current score", "The current volume", "Whether a key was pressed", "Whether one sprite is overlapping/touching another sprite or the mouse-pointer"],
        correct: 3,
        why: "The 'touching' block checks if a sprite is currently overlapping another sprite, the mouse-pointer, or an edge, often used for collisions."
      },
      {
        q: "Can a single Scratch project have more than one sprite, each with its own scripts?",
        a: ["Only two sprites are allowed maximum", "No, only one sprite is allowed per project", "Yes, and each sprite's scripts run independently based on their own events", "Sprites must always share the exact same script"],
        correct: 2,
        why: "Scratch projects can contain many sprites, each running its own independent scripts triggered by its own events."
      },
      {
        q: "What does 'clone' do in Scratch?",
        a: ["Creates a temporary copy of a sprite that can run its own scripts", "Stops all scripts", "Deletes the original sprite", "Changes the background"],
        correct: 0,
        why: "Cloning creates a duplicate of a sprite at runtime, useful for things like spawning multiple enemies or bullets that each run independently."
      },
      {
        q: "Why does resetting a variable at the start of a game (like 'set lives to 3') matter?",
        a: ["Only affects background music", "It always breaks the game", "Without resetting, the value could carry over incorrectly from a previous run", "It has no effect on gameplay"],
        correct: 2,
        why: "Resetting a variable ensures the game starts in a known, correct state instead of carrying over leftover values from a prior play session."
      }
    ]
  },
  {
    id: "recommend",
    day: 3,
    title: "Python Basics",
    emoji: "\ud83d\udc0d",
    color: "#d89e00",
    subtitle: "print(), for loops, turtle graphics & variables",
    questions: [
      {
        q: "In Python, what does print(\"Hello, World!\") do?",
        a: ["Creates a variable named Hello", "Deletes a file", "Draws a shape", "Displays the text to the screen"],
        correct: 3,
        why: "print() is a built-in function that displays text to the screen \u2014 the program's way of talking back to you."
      },
      {
        q: "Why does print(Hello) cause an error, but print(\"Hello\") works?",
        a: ["Hello is a banned word", "print() can only be used once per program", "Without quotes, Python thinks Hello is a variable name, not text", "print() needs a number instead"],
        correct: 2,
        why: "Text must be wrapped in quotes to become a string. Without quotes, Python looks for a variable called Hello and won't find one."
      },
      {
        q: "What does 'for i in range(5):' do in Python?",
        a: ["Draws a 5-sided shape automatically", "Prints the number 5", "Creates 5 variables", "Repeats the indented code 5 times, with i counting 0,1,2,3,4"],
        correct: 3,
        why: "range(5) gives the numbers 0 through 4 (5 total), and the loop runs once for each of them."
      },
      {
        q: "In Python, what tells the computer which lines belong inside a for loop?",
        a: ["Capital letters", "Indentation (spaces at the start of the line)", "The color of the text", "Question marks"],
        correct: 1,
        why: "Python uses indentation (usually 4 spaces) to know what code is inside the loop. In Scratch, block-snapping does this automatically."
      },
      {
        q: "What must every 'for' line end with in Python?",
        a: ["A period", "A colon ( : )", "A semicolon", "Nothing special"],
        correct: 1,
        why: "Every for line ends with a colon \u2014 Python's way of saying 'here comes the indented block.'"
      },
      {
        q: "In the turtle spiral code 'for i in range(100): t.forward(i)', why does the spiral grow bigger each time?",
        a: ["forward() always draws the same length", "The variable i increases each loop, so forward(i) moves farther every step", "It doesn't actually grow", "The turtle gets tired"],
        correct: 1,
        why: "i starts at 0 and increases each time through the loop, so t.forward(i) makes each line segment longer \u2014 creating a spiral."
      },
      {
        q: "What command is needed at the very end of a turtle program to keep the drawing window open?",
        a: ["turtle.done()", "turtle.close()", "turtle.exit()", "turtle.finish()"],
        correct: 0,
        why: "Without turtle.done() as the last line, the window closes the instant the drawing finishes."
      },
      {
        q: "How does using random.choice() with a turtle spiral relate to generative AI?",
        a: ["Random turtle art always looks identical each run", "Generative AI similarly mixes learned rules with randomness so each output is unique", "AI never uses randomness", "It has no connection to AI"],
        correct: 1,
        why: "Generative AI follows patterns it learned from data, and adds randomness \u2014 so each run is unique, just like the turtle spiral."
      },
      {
        q: "In Python, what symbol assigns a value to a variable?",
        a: ["A single equals sign ( = )", "A colon ( : )", "A double equals sign ( == )", "A question mark ( ? )"],
        correct: 0,
        why: "A single = stores/assigns a value into a variable, like name = \"Alex\". Double == is used later for comparisons."
      },
      {
        q: "What does input() always return in Python?",
        a: ["Nothing", "A string (text) — even if the user types a number", "A whole number", "A boolean"],
        correct: 1,
        why: "input() always returns a string. To use it as a number, you must convert it, e.g. with int(input(...))."
      },
      {
        q: "What is an f-string used for, like f\"My name is {name}\"?",
        a: ["It runs a loop", "It draws with turtle", "It formats and fills in variable values directly inside a string", "It deletes a variable"],
        correct: 2,
        why: "The f before the quotes tells Python to look for {} and fill them in with real variable values \u2014 the most readable way to build output."
      }
    ,
      {
        q: "What does the len() function do in Python?",
        a: ["Returns the number of items in a list or characters in a string", "Converts text to numbers", "Deletes an item from a list", "Adds a new item to a list"],
        correct: 0,
        why: "len() returns the count of items in a sequence, whether that's a list, string, or other collection."
      },
      {
        q: "What data type does 3.14 belong to in Python?",
        a: ["Boolean", "Float (a decimal number)", "Integer", "String"],
        correct: 1,
        why: "Numbers with a decimal point, like 3.14, are floats in Python, distinct from whole-number integers."
      },
      {
        q: "What is the correct way to write a comment in Python?",
        a: ["// This is a comment", "# This is a comment", "<!-- This is a comment -->", "/* This is a comment */"],
        correct: 1,
        why: "Python comments start with a # symbol; everything after it on that line is ignored by the interpreter."
      },
      {
        q: "What does str(5) return in Python?",
        a: ["An error", "The integer 5", "The boolean True", "The text string \"5\""],
        correct: 3,
        why: "str() converts a value into its text (string) representation, turning the number 5 into the string \"5\"."
      },
      {
        q: "In turtle graphics, what does t.right(90) do?",
        a: ["Moves the turtle forward 90 steps", "Changes the pen color", "Turns the turtle 90 degrees counter-clockwise", "Turns the turtle's heading 90 degrees clockwise"],
        correct: 3,
        why: "t.right(90) rotates the turtle's current heading by 90 degrees in the clockwise direction, without moving it forward."
      },
      {
        q: "What does t.penup() do in turtle graphics?",
        a: ["Lifts the pen so the turtle moves without drawing a line", "Draws a line immediately", "Deletes the drawing", "Changes the turtle's speed"],
        correct: 0,
        why: "penup() lifts the pen so subsequent movement commands move the turtle without leaving a trail, until pendown() is called again."
      },
      {
        q: "Why do we call print() a 'function' in Python?",
        a: ["It's a named, reusable block of code that performs a specific action when called", "print() only works once per program", "It's not really a function, just a keyword", "Functions only exist in Scratch"],
        correct: 0,
        why: "print() is a built-in function: a named block of reusable code that performs an action (displaying text) whenever it's called."
      },
      {
        q: "What will 'for i in range(3):' followed by an indented print(i) output?",
        a: ["0 1 2, one per line", "1 2 3, one per line", "3 3 3", "Nothing, this causes an error"],
        correct: 0,
        why: "range(3) produces 0, 1, 2 and the loop prints each value once, resulting in three lines: 0, 1, and 2."
      },
      {
        q: "What is the purpose of variables like name = \"Alex\" in a Python program?",
        a: ["They immediately print to the screen", "They delete previous variables", "They only work with numbers", "They store a value under a label so it can be reused and changed later"],
        correct: 3,
        why: "Variables label and store values in memory so the program can reuse, reference, or update them later in the code."
      },
      {
        q: "Why might a beginner's turtle program draw nothing visible on screen?",
        a: ["Common causes include the pen being up, wrong coordinates, or missing turtle.done()", "Turtle graphics never works reliably", "Python cannot draw shapes at all", "The screen resolution is too low"],
        correct: 0,
        why: "Common turtle mistakes include forgetting pendown(), using incorrect coordinates, or omitting turtle.done() to keep the window open."
      },
      {
        q: "What is the result of combining two strings with + in Python, e.g. \"Hi \" + \"there\"?",
        a: ["An error, since + only works on numbers", "\"Hi\" and \"there\" as separate values", "\"Hi there\", the two strings joined together", "The number 0"],
        correct: 2,
        why: "The + operator concatenates (joins) strings together, so \"Hi \" + \"there\" produces the combined string \"Hi there\"."
      },
      {
        q: "In Python, why is int(input(\"Enter age: \")) commonly used?",
        a: ["int() deletes whatever was typed", "input() always returns text, so int() converts the typed age into a usable number", "It has no real effect", "input() already returns numbers, so int() is unnecessary"],
        correct: 1,
        why: "Since input() always returns a string, wrapping it in int() converts the text into an actual number for calculations or comparisons."
      },
      {
        q: "What does the turtle module let you control that print() does not?",
        a: ["Loop counting", "Visual drawing on screen, like lines and shapes via movement commands", "Variable creation", "Text output only"],
        correct: 1,
        why: "The turtle module provides tools to draw visual graphics on screen by moving a virtual 'turtle,' something print() cannot do since it only outputs text."
      },
      {
        q: "How does f-string formatting improve readability compared to using + to join strings and variables?",
        a: ["f-strings run code twice", "It lets you embed variables directly inside {} within the string, avoiding messy concatenation", "It doesn't improve readability at all", "f-strings can only be used with numbers"],
        correct: 1,
        why: "f-strings let you place variables directly inside {} in the text, making the code far more readable than chains of + concatenation."
      }
    ]
  },
  {
    id: "images",
    day: 4,
    title: "Python: Logic & Games",
    emoji: "\ud83c\udfae",
    color: "#9b5de5",
    subtitle: "if/elif/else, random, lists & building real games",
    questions: [
      {
        q: "In Python, what is the difference between = and ==?",
        a: ["= stores a value; == compares two values for equality", "== stores a value; = compares values", "They are the same thing", "Neither can be used in Python"],
        correct: 0,
        why: "A single = assigns/stores a value. A double == checks whether two values are equal, returning True or False."
      },
      {
        q: "What does random.randint(1, 100) do?",
        a: ["Returns one random whole number between 1 and 100", "Deletes a variable", "Always returns 1", "Returns a list of 100 numbers"],
        correct: 0,
        why: "random.randint(1, 100) hands back one random integer in that range every time it's called."
      },
      {
        q: "In the number-guessing game, why does 'higher/lower' feedback help find the number quickly?",
        a: ["It doesn't help at all", "It always takes 100 guesses", "Each answer cuts the remaining search space roughly in half — a strategy called binary search", "It only works for even numbers"],
        correct: 2,
        why: "This is binary search \u2014 repeatedly cutting the range in half. It can find a number 1\u2013100 in about 7 guesses instead of 100."
      },
      {
        q: "In Python's if/elif/else, what happens after the first true condition is found?",
        a: ["Python throws an error", "Python runs that branch and skips the rest", "Nothing runs", "Python still checks every remaining condition"],
        correct: 1,
        why: "Python checks conditions top to bottom and runs the first one that's true, then skips the rest \u2014 it does not keep checking."
      },
      {
        q: "Why must you wrap input() with int() before comparing it to a number, like int(input(...)) < 50?",
        a: ["It's optional and does nothing", "input() always returns text, and comparing text to a number causes an error", "int() deletes the input", "Python requires it only for negative numbers"],
        correct: 1,
        why: "input() always returns a string. Wrapping it with int() converts the text into a real number so it can be compared."
      },
      {
        q: "What is a list used for in Python, as introduced in the Rock-Paper-Scissors lesson?",
        a: ["Storing a single number", "Drawing shapes", "Only for turtle graphics", "Organizing multiple choices or items in one place, like [\"rock\",\"paper\",\"scissors\"]"],
        correct: 3,
        why: "A list stores many items together in one place \u2014 perfect for holding the set of choices in a game like Rock-Paper-Scissors."
      },
      {
        q: "How can a computer 'randomly choose' an item from a list in Python?",
        a: ["random.choice(list)", "list.delete()", "print(list)", "input(list)"],
        correct: 0,
        why: "random.choice(list) picks one random item from the list \u2014 used to make the computer 'pick' rock, paper, or scissors."
      },
      {
        q: "What does a 'while True:' loop do if you forget to add a break statement?",
        a: ["It causes a syntax error", "It only runs once", "It stops automatically after 10 times", "It runs forever, since there's no exit condition"],
        correct: 3,
        why: "A while True: loop has no built-in stopping point \u2014 you must add a break inside it (e.g., when the guess is correct) to end it."
      },
      {
        q: "What is the purpose of a function like get_reply(msg) in a Python program?",
        a: ["Functions serve no purpose", "To permanently delete msg", "To slow the program down", "To package reusable logic into one named block you can call again and again"],
        correct: 3,
        why: "A function lets you write one block of logic once and call it repeatedly, instead of copy-pasting the same code over and over."
      },
      {
        q: "What is a syntax error in Python?",
        a: ["A perfectly normal, correct program", "An error that only happens with numbers", "The program runs but gives the wrong answer", "The computer can't even understand the code — like a typo or missing colon — so it won't run at all"],
        correct: 3,
        why: "A syntax error means Python can't understand the code at all \u2014 for example a missing colon. The program won't run until it's fixed."
      },
      {
        q: "What is a logic error in Python?",
        a: ["The program crashes immediately with an error message", "It's the same as a syntax error", "It only happens in Scratch", "The program runs without crashing but does the wrong thing, like adding instead of subtracting"],
        correct: 3,
        why: "A logic error is the hardest to find \u2014 there's no error message, the program just quietly does the wrong thing."
      }
    ,
      {
        q: "What does 'elif' stand for in Python?",
        a: ["Else input file", "End loop if", "Else if — an additional condition checked if the prior ones were false", "Extra list if function"],
        correct: 2,
        why: "'elif' is short for 'else if' — it checks another condition only if the earlier if/elif conditions were false."
      },
      {
        q: "What will random.choice([\"a\",\"b\",\"c\"]) return?",
        a: ["An error, since choice() needs numbers", "One randomly selected item from the list", "All three items joined together", "Always \"a\""],
        correct: 1,
        why: "random.choice() picks and returns a single random item from the given list each time it is called."
      },
      {
        q: "In a Rock-Paper-Scissors game, what happens if the player and computer pick the same option?",
        a: ["The computer automatically wins", "The player automatically loses", "The game crashes", "It's typically treated as a tie/draw"],
        correct: 3,
        why: "When both players choose the same option in Rock-Paper-Scissors, the standard result is a tie, with neither side winning that round."
      },
      {
        q: "What is the purpose of a 'break' statement inside a loop?",
        a: ["It immediately exits the loop, skipping any remaining iterations", "It restarts the loop from the beginning", "It has no effect inside loops", "It pauses the loop for one second"],
        correct: 0,
        why: "'break' immediately stops the loop's execution and exits it, even if the loop condition would otherwise continue running."
      },
      {
        q: "Why might a program raise an 'IndexError' when accessing a list?",
        a: ["IndexError never happens in Python", "The list is too colorful", "Lists cannot have errors", "The code tried to access an index that doesn't exist in the list, like item 10 in a 3-item list"],
        correct: 3,
        why: "An IndexError occurs when code tries to access a position in a list that is out of range, such as index 10 in a list with only 3 items."
      },
      {
        q: "What does list.append(item) do?",
        a: ["Deletes the entire list", "Adds a new item to the end of the list", "Sorts the entire list", "Removes the last item from the list"],
        correct: 1,
        why: "append() adds a new item onto the end of an existing list, growing its length by one."
      },
      {
        q: "In Python, what does the modulo operator % do, as in 10 % 3?",
        a: ["Multiplies 10 and 3", "Always returns 0", "Divides 10 by 3 and returns the remainder (1)", "Converts numbers to percentages"],
        correct: 2,
        why: "The modulo operator % returns the remainder after division, so 10 % 3 equals 1 because 3 goes into 10 three times with 1 left over."
      },
      {
        q: "Why is a logic error often harder to find than a syntax error?",
        a: ["Python fixes logic errors automatically", "Logic errors always crash the program with a clear message", "A logic error produces no error message; the program just quietly gives the wrong answer", "Logic errors are actually easier to find"],
        correct: 2,
        why: "Unlike a syntax error, a logic error runs without crashing, silently producing a wrong result, which makes it much harder to detect."
      },
      {
        q: "What is the benefit of writing a function like check_guess(num) instead of repeating the same comparison code many times?",
        a: ["It lets you reuse the same logic by calling the function whenever needed, keeping code organized", "Functions can only be called once", "There is no benefit to using functions", "Functions make code longer for no reason"],
        correct: 0,
        why: "Functions package logic into a reusable, named block, so you can call it repeatedly without duplicating the same code throughout your program."
      },
      {
        q: "What does 'while guess != target:' mean in Python?",
        a: ["This is invalid syntax", "Keep looping as long as guess is NOT equal to target", "Loop exactly once", "Loop forever, no matter what"],
        correct: 1,
        why: "!= means 'not equal to,' so the while loop continues running as long as guess and target are different values."
      },
      {
        q: "Why should you avoid deeply nested if statements when possible?",
        a: ["Excessive nesting makes code harder to read and debug; simpler structures or functions are often clearer", "Python doesn't allow nested ifs", "Nested ifs run twice as slow always", "Nesting always causes crashes"],
        correct: 0,
        why: "Too many nested if statements make code visually complex and harder to trace, so simplifying logic or breaking it into functions often helps readability."
      },
      {
        q: "What is the result of bool(0) in Python?",
        a: ["True", "An error", "0", "False"],
        correct: 3,
        why: "In Python, the number 0 is considered 'falsy,' so bool(0) evaluates to False."
      },
      {
        q: "Why is choosing meaningful list contents important in a Rock-Paper-Scissors game, like choices = [\"rock\",\"paper\",\"scissors\"]?",
        a: ["It doesn't matter what's in the list", "Lists can only contain numbers", "The list order changes the game's rules", "The list defines every valid option the game can randomly select from"],
        correct: 3,
        why: "The list defines the complete set of valid choices the game logic and random.choice() can select from during play."
      },
      {
        q: "What does list.sort() do to a Python list?",
        a: ["Adds a new item", "Deletes all items", "Reverses the list only", "Rearranges the items into ascending order in place"],
        correct: 3,
        why: "sort() rearranges the list's items into ascending order (or by a custom rule), modifying the list directly."
      }
    ]
  },
  {
    id: "chatbot",
    day: 5,
    title: "AI in Practice",
    emoji: "\ud83e\udd16",
    color: "#00bbf9",
    subtitle: "Teachable Machine, chatbots, bias & fairness",
    questions: [
      {
        q: "In Teachable Machine, what is 'training data'?",
        a: ["A type of computer virus", "The labeled examples (pictures, sounds, or poses) you show the computer to learn from", "The name of the app", "The final answer the model gives"],
        correct: 1,
        why: "Training data is the set of labeled examples the model learns from \u2014 like pictures labeled 'Thumbs Up' or 'Thumbs Down.'"
      },
      {
        q: "In machine learning, what is a 'model'?",
        a: ["A photograph", "The pattern the computer builds by comparing all the training examples — its 'trained brain' for one task", "A Scratch sprite", "A type of variable"],
        correct: 1,
        why: "The model is the pattern the computer builds after seeing all the training examples \u2014 its trained brain for that specific task."
      },
      {
        q: "In machine learning, what is a 'prediction'?",
        a: ["A type of bug", "A random number", "The training data itself", "The model's best guess on something new, along with a confidence score"],
        correct: 3,
        why: "A prediction is the model's guess about something new it hasn't seen before, plus how confident it is in that guess."
      },
      {
        q: "Why might a model trained on only 2-3 examples per class give unreliable predictions?",
        a: ["Small amounts of data can't cover every angle, lighting, or variation, so the model makes more mistakes on new cases", "More examples always make a model worse", "The model needs exactly 2 examples to work perfectly", "Training data doesn't affect predictions"],
        correct: 0,
        why: "Too few or too similar examples mean the model only really knows those specific cases \u2014 not the general concept."
      },
      {
        q: "What does the word 'bias' mean when talking about AI training data?",
        a: ["When the AI runs slowly", "When a model uses too much memory", "When limited or unfair data makes a model work poorly for some people or cases", "When a program has a syntax error"],
        correct: 2,
        why: "Bias happens when the training data doesn't fairly represent everyone, so the model performs worse for people or cases it wasn't trained on."
      },
      {
        q: "Who is responsible when an AI model turns out to be biased or unfair?",
        a: ["Only the original inventor of computers", "No one is ever responsible", "The people who chose what data to feed it — that's a decision, not an accident", "The computer itself, since it decided on its own"],
        correct: 2,
        why: "Bias comes from choices about what data to use. The people building and training the model are responsible for those choices."
      },
      {
        q: "What is a 'rule-based' chatbot?",
        a: ["A chatbot that checks your message for keywords and picks a pre-written reply that a human wrote", "A chatbot that only works with images", "The same thing as a large language model", "A chatbot that learns entirely on its own with no help"],
        correct: 0,
        why: "A rule-based chatbot follows rules humans wrote in advance \u2014 'if the message contains hello, say hi back.' No learning happens."
      },
      {
        q: "Why is .lower() important in a keyword-matching chatbot?",
        a: ["It deletes the user's message", "It makes the program run faster", "It has no real purpose", "It converts text to lowercase so 'Hello', 'HELLO', and 'hello' all match the same keyword rule"],
        correct: 3,
        why: "Without .lower(), the chatbot would only match exact capitalization, missing 'Hello' if it was only checking for 'hello'."
      },
      {
        q: "How is a rule-based chatbot fundamentally different from a real large language model (LLM) like ChatGPT?",
        a: ["LLMs also only use pre-written rules", "Rule-based bots are smarter than LLMs", "The rule-based bot can only say what a human explicitly wrote; an LLM predicts new text from patterns learned in billions of examples", "There is no real difference"],
        correct: 2,
        why: "A rule-based bot is fully explainable and limited to pre-written replies. An LLM was trained on patterns and can generate text nobody explicitly wrote."
      },
      {
        q: "In the chatbot lesson, what does an 'else' branch handle?",
        a: ["Cases where no keyword rule matched, so the bot needs a fallback reply like 'I don't understand'", "It deletes the conversation", "It only runs at the start of the program", "The very first message only"],
        correct: 0,
        why: "The else branch is the fallback for when nothing else matched \u2014 giving the chatbot a friendly reply instead of crashing or staying silent."
      },
      {
        q: "What is the purpose of a fairness/clarity check on an AI project, like checking if it's 'FAIR, CLEAR, TESTED'?",
        a: ["It's only for graphics quality", "To count the lines of code", "To confirm the AI works for people beyond just the examples tested, and that users understand it's an AI, not magic", "To make the code run faster"],
        correct: 2,
        why: "A fairness and clarity check asks whether the AI works broadly (not just for tested cases) and whether it's clear to users that it's an AI system making decisions."
      }
    ,
      {
        q: "What is 'overfitting' in a simple machine learning model?",
        a: ["The model memorizes training examples too closely and performs poorly on new, unseen examples", "Overfitting means the model learned perfectly for all situations", "The model is too small to store any data", "The model always guesses correctly"],
        correct: 0,
        why: "Overfitting happens when a model learns the training examples too specifically, so it fails to generalize well to new, unseen data."
      },
      {
        q: "Why is having a diverse set of training examples important for a Teachable Machine model?",
        a: ["Diversity makes training slower for no benefit", "Diverse examples help the model recognize the concept across different conditions instead of just the exact training photos", "Diversity is not important at all", "Only one example is ever needed"],
        correct: 1,
        why: "Diverse training examples covering different angles, lighting, and variations help the model generalize rather than only recognizing the exact training images."
      },
      {
        q: "What does a 'confidence score' from an AI model represent?",
        a: ["How sure the model is about its prediction, often shown as a percentage", "The amount of training data used", "The speed of the model", "The exact correct answer, guaranteed"],
        correct: 0,
        why: "A confidence score shows how sure the model is in its prediction, typically as a percentage — it's not a guarantee of correctness."
      },
      {
        q: "How can a chatbot handle a misspelled keyword, like 'helo' instead of 'hello'?",
        a: ["Rule-based bots always ignore text entirely", "Misspellings crash all chatbots", "A rule-based bot generally won't recognize it unless a rule was written for that specific misspelling", "It automatically understands all misspellings perfectly"],
        correct: 2,
        why: "A simple rule-based chatbot only matches keywords it was explicitly programmed to look for, so unanticipated misspellings often go unrecognized."
      },
      {
        q: "What is one advantage of a rule-based chatbot over a large language model?",
        a: ["It never needs any code", "It's more powerful than any LLM", "It can learn new topics on its own", "Its responses are fully predictable and explainable, since a human wrote every possible reply"],
        correct: 3,
        why: "Rule-based chatbots are fully predictable and explainable because every possible response was explicitly written by a human in advance."
      },
      {
        q: "Why do modern AI image classifiers need thousands of training images rather than just a handful?",
        a: ["More examples help the model learn general patterns rather than memorizing a few specific images", "Thousands of images always cause errors", "One image is always sufficient for any task", "More images always slow the model down with no benefit"],
        correct: 0,
        why: "Larger, more varied training sets help models learn general, reliable patterns instead of just memorizing a tiny number of specific examples."
      },
      {
        q: "What ethical issue can arise if a face-recognition model is trained mostly on one skin tone?",
        a: ["It automatically fixes itself over time", "The model will refuse to run", "There is no ethical issue at all", "It may perform less accurately on people with different skin tones, which is an example of biased training data"],
        correct: 3,
        why: "Training data that lacks diversity can cause a model to perform worse for underrepresented groups, a real-world example of AI bias."
      },
      {
        q: "What does 'garbage in, garbage out' mean in the context of AI training data?",
        a: ["If the training data is flawed or low-quality, the model's output will also be flawed", "It means the model deletes all bad data automatically", "AI can turn bad data into perfect results", "It refers to recycling old code"],
        correct: 0,
        why: "'Garbage in, garbage out' means that poor-quality or biased training data leads directly to poor-quality or biased model outputs."
      },
      {
        q: "Why should chatbot developers test their bot with many different phrasings of the same question?",
        a: ["Testing multiple phrasings is unnecessary busywork", "Users won't always phrase things the same way, so testing multiple phrasings reveals gaps in keyword coverage", "Chatbots automatically understand all phrasings", "One phrasing test always covers every case"],
        correct: 1,
        why: "Real users type things differently, so testing several phrasings helps developers catch missing keyword rules and improve the bot's coverage."
      },
      {
        q: "What is a simple way to give a chatbot a 'personality'?",
        a: ["Randomizing every single word it says", "Personality is impossible to add to any chatbot", "Personality only comes from machine learning, never rules", "Writing consistent tone and style into its pre-written replies, like being cheerful or formal"],
        correct: 3,
        why: "A chatbot's personality is often just intentional, consistent word choice and tone baked into its pre-written responses by the developer."
      },
      {
        q: "Why is it misleading to say an AI model 'understands' language the way a human does?",
        a: ["AI always explains its reasoning in full detail", "AI predicts likely word patterns from data, without true comprehension or consciousness like a human has", "It's not misleading, AI understands exactly like humans", "Humans also just predict word patterns"],
        correct: 1,
        why: "AI models predict statistically likely word patterns from training data; they don't have true understanding, consciousness, or comprehension the way humans do."
      },
      {
        q: "What is a good habit when evaluating any AI tool's output, according to responsible AI use?",
        a: ["Double-check important facts, since AI can sound confident while being wrong", "Never use AI tools for anything", "Trust it completely without question", "Assume AI is always up to date"],
        correct: 0,
        why: "Because AI can confidently state incorrect information, it's a good habit to verify important facts rather than trusting AI output blindly."
      },
      {
        q: "Why might a chatbot give an unhelpful reply to a perfectly reasonable question?",
        a: ["Unhelpful replies never happen", "A rule-based bot may lack a matching keyword rule, or an AI model may lack relevant training data for that topic", "Reasonable questions always break chatbots", "Chatbots are always perfect"],
        correct: 1,
        why: "Unhelpful replies often occur because a rule-based bot has no matching rule, or a trained model has too little relevant data on that topic."
      },
      {
        q: "What is the purpose of labeling images 'thumbs up' or 'thumbs down' before training a classifier?",
        a: ["Labels teach the model which category each example belongs to, so it can learn to recognize the difference", "Labels only matter after training is finished", "Labels are decorative and don't affect training", "Labels replace the need for any images"],
        correct: 0,
        why: "Labels tell the model exactly which category each training example belongs to, which is essential for it to learn the distinction between classes."
      }
    ]
  },
  {
    id: "bias",
    day: 6,
    title: "Capstone & Shipping Code",
    emoji: "\ud83d\ude80",
    color: "#ef476f",
    subtitle: "Planning, debugging, and shipping a real project",
    questions: [
      {
        q: "What must every capstone project include, according to the Capstone Kickoff lesson?",
        a: ["At least 100 lines of code", "One AI idea that the student can point to and explain why it counts as AI", "A turtle graphics drawing", "A high score system"],
        correct: 1,
        why: "Every capstone must include or explain one AI idea, whether it's a trained classifier, a keyword chatbot, or rule-based 'AI' character."
      },
      {
        q: "In the capstone planning sheet, what four things does a good project plan define?",
        a: ["Colors, fonts, sounds, and animations", "The teacher's name and the date", "Only the programming language used", "What it does, who it's for, which concepts it uses, and what 'done' looks like"],
        correct: 3,
        why: "A clear project plan answers: what does it do, who is it for, which coding concepts will it use, and what does 'done' look like."
      },
      {
        q: "What is the recommended debugging method taught in the capstone build sessions?",
        a: ["Randomly change things until it works", "Stop → Check (what should happen vs. what is happening) → Find → Fix one thing at a time → Test", "Ignore bugs and keep adding features", "Delete the whole project and start over"],
        correct: 1,
        why: "The method is: Stop, Check what should happen vs. what's actually happening, Find the line, Fix one thing at a time, then Test again."
      },
      {
        q: "Why should you change only ONE thing at a time while debugging?",
        a: ["So you know exactly which change fixed (or broke) something, instead of guessing", "Multiple changes always work better", "It makes debugging take longer", "Python requires it by law"],
        correct: 0,
        why: "Changing one thing at a time lets you clearly identify which specific change fixed the problem \u2014 random multi-changes make it unclear what worked."
      },
      {
        q: "What is a good debugging trick if you're not sure what value a variable currently holds?",
        a: ["Delete the variable", "Guess and hope for the best", "Have the program display or print the variable's value so you can see it directly", "Rename the variable"],
        correct: 2,
        why: "Displaying a variable (with a 'say' block in Scratch or print() in Python) lets you see exactly what value it holds at that point."
      },
      {
        q: "What is 'Feature Freeze' in the capstone timeline?",
        a: ["The last day of the entire course", "A day to add lots of new features", "A stage where you stop adding new ideas and focus on making existing features work reliably every time", "A type of bug"],
        correct: 2,
        why: "Feature Freeze means no new features \u2014 just making sure everything you've already built actually works every single time."
      },
      {
        q: "According to the showcase rehearsal lesson, what should a strong project demo do in the first 20 seconds?",
        a: ["Show only the bugs", "Play loud music", "List every line of code", "Make it clear to a first-time viewer what the project does"],
        correct: 3,
        why: "A strong demo makes it immediately clear what the project does, even to someone seeing it for the very first time."
      },
      {
        q: "Is it better for a demo to only show perfect parts, or also explain a bug still being fixed?",
        a: ["Demos should have no explanation at all", "Only show perfect parts, hide all problems", "It's stronger to also explain a known bug — it shows real understanding of the code, not just clicking buttons", "Bugs should never be mentioned"],
        correct: 2,
        why: "Explaining a known bug shows you understand your own code deeply \u2014 proving comprehension, not just button-clicking."
      },
      {
        q: "Why do professional engineers spend so much time debugging?",
        a: ["Only beginners need to debug", "Professional coders at companies like Google and Apple spend about half their time debugging — it's a normal, expected part of the job", "Debugging means the code is unfixable", "Debugging is rare and unimportant"],
        correct: 1,
        why: "Professional coders spend roughly 50% of their time debugging. Bugs aren't failures \u2014 they're a normal and expected part of building software."
      },
      {
        q: "What is the difference between a runtime error and a logic error?",
        a: ["Runtime errors never happen in Python", "They are exactly the same thing", "A logic error always crashes the program", "A runtime error crashes the program while running; a logic error runs fine but produces the wrong result silently"],
        correct: 3,
        why: "A runtime error stops the program suddenly while running (like dividing by zero). A logic error runs fine but quietly produces the wrong answer \u2014 no error message appears."
      },
      {
        q: "What is the main goal of 'Capstone Build Day' sessions?",
        a: ["Getting the smallest real, working slice of the project running early", "Only watching videos", "Deleting all previous code", "Skipping straight to the final showcase"],
        correct: 0,
        why: "Capstone build days focus on getting a first working slice running early, rather than waiting until the last minute to test anything."
      }
    ,
      {
        q: "Why is it helpful to write out your capstone idea in a single sentence before coding?",
        a: ["It forces clarity about what you're building before getting lost in code details", "It's not helpful at all", "One-sentence plans are always wrong", "Only teachers need written plans"],
        correct: 0,
        why: "Summarizing your idea in one clear sentence forces you to clarify the goal before diving into code, preventing wasted effort on unclear plans."
      },
      {
        q: "What is a 'minimum viable version' of a capstone project?",
        a: ["A broken version with no working parts", "The final, fully polished version with every feature", "A version with unnecessary extra features only", "The absolute smallest working version that demonstrates the core idea"],
        correct: 3,
        why: "A minimum viable version is the smallest possible version that still demonstrates the core idea working, which you can then build on."
      },
      {
        q: "Why is it useful to ask a friend to test your project before the showcase?",
        a: ["A tester unfamiliar with your code often finds confusing parts or bugs you missed because you're used to it", "Only the original coder should ever test a project", "Friends can't give useful feedback", "Testing wastes valuable time"],
        correct: 0,
        why: "Someone unfamiliar with your code brings a fresh perspective and often notices confusing parts or bugs that you've become blind to after working on it."
      },
      {
        q: "What should you do if you run out of time before finishing every planned capstone feature?",
        a: ["Delete the whole project", "Add even more unfinished features", "Panic and submit nothing", "Prioritize showing your strongest, most complete features rather than several unfinished ones"],
        correct: 3,
        why: "When time runs short, it's better to polish and present your strongest working features than to show many incomplete, broken ones."
      },
      {
        q: "Why is version control (like saving copies of your code) useful while building a project?",
        a: ["It automatically writes your code for you", "It has no real benefit", "It prevents all bugs from ever happening", "It lets you go back to a working version if a new change breaks something"],
        correct: 3,
        why: "Keeping saved copies or versions of your code lets you revert to a known-working state if a new change accidentally introduces a bug."
      },
      {
        q: "What is the value of explaining your code's logic out loud to someone else, even a rubber duck?",
        a: ["It always wastes time better spent coding", "Talking through your logic can help you notice gaps or mistakes you missed while just reading silently", "Explaining code out loud has no benefit", "It only helps if the listener is an expert"],
        correct: 1,
        why: "Explaining your logic aloud (even to an inanimate object, a classic technique called 'rubber duck debugging') often reveals gaps in reasoning you missed while reading silently."
      },
      {
        q: "Why should you comment your code with brief notes explaining tricky parts?",
        a: ["Comments are required by Python syntax", "Comments replace the need for testing", "Comments help you (and others) remember why code was written a certain way when you return to it later", "Comments slow down the computer significantly"],
        correct: 2,
        why: "Brief comments help you and others quickly recall the purpose of tricky code sections later, saving time during debugging or future edits."
      },
      {
        q: "What is the danger of adding many new features right before a deadline?",
        a: ["Last-minute changes are more likely to introduce new bugs with little time left to test them", "Deadlines don't affect code quality", "There is no danger at all", "New features always work perfectly with no risk"],
        correct: 0,
        why: "Adding features close to a deadline increases the risk of new bugs, with little remaining time to properly test and fix them — hence 'Feature Freeze.'"
      },
      {
        q: "Why is it good practice to save your project frequently while working?",
        a: ["It protects your progress in case of a crash, accidental deletion, or unexpected error", "Saving is only necessary once, at the very end", "Saving automatically fixes bugs", "Frequent saving wastes storage space with no benefit"],
        correct: 0,
        why: "Frequent saving protects your progress against crashes, accidental deletions, or unexpected errors, preventing you from losing significant work."
      },
      {
        q: "What makes a project demo confusing to a first-time viewer?",
        a: ["Explaining basic functionality clearly at the start", "Showing the main feature early", "Jumping straight into small details without first explaining what the project even does", "Speaking clearly and slowly"],
        correct: 2,
        why: "Diving into small details before explaining the project's overall purpose leaves first-time viewers confused about what they're even watching."
      },
      {
        q: "Why might a programmer intentionally leave a known minor bug unfixed before a showcase deadline?",
        a: ["Programmers are lazy and don't care", "Bugs should never be left unfixed under any circumstance", "All bugs are equally critical to fix immediately", "If fixing it risks breaking other working features with limited time left, it may be safer to document it and move on"],
        correct: 3,
        why: "With limited time, risky fixes can break other working parts of the project; sometimes it's wiser to document a minor known bug and preserve stability."
      },
      {
        q: "What is the benefit of practicing your project demo out loud before the actual showcase?",
        a: ["Practice always makes performances worse", "Practicing has no real benefit", "Only written demos need practice", "It helps you notice awkward explanations, timing issues, or bugs you might otherwise miss during the live demo"],
        correct: 3,
        why: "Rehearsing out loud helps catch awkward wording, pacing issues, or bugs that might not be obvious until you actually try presenting the project."
      },
      {
        q: "What is the benefit of breaking a big capstone idea into smaller milestones?",
        a: ["Smaller milestones make progress trackable and prevent feeling overwhelmed by the whole project at once", "Breaking down ideas is unnecessary for small projects", "Milestones only slow down real progress", "Milestones replace the need for a final project"],
        correct: 0,
        why: "Breaking a big idea into smaller milestones makes progress easier to track and prevents the overwhelm of tackling everything at once."
      },
      {
        q: "Why is it valuable to keep a simple to-do list while building your capstone?",
        a: ["To-do lists are only useful for chores, not code", "Only teachers need to-do lists", "It helps you track what's done, what's next, and what still needs testing, reducing forgotten steps", "To-do lists slow down programmers"],
        correct: 2,
        why: "A to-do list helps track completed work, upcoming tasks, and pending tests, reducing the chance that important steps get forgotten."
      }
    ]
  }
];
