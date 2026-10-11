# One question, two task settings

[Documentation](README.md) · [Experiments](EXPERIMENTS.md) · [Run a policy](QUICKSTART.md)

The project began with phone videos of a hand picking up a small black block. The aim was to find out which information in those recordings could be useful to a robot. A Panda simulation supplied repeatable robot trials and contact-based scoring.

## Start with a simple transfer

The first task was to lift the block and place it beyond a black line. Its fixed controller separated two problems: estimating the block's position and executing a grasp. This let the experiment change the visual training targets while keeping the robot controller the same.

Twelve personal clips supplied real images. Four visual groups learned with RGB consistency alone, object-box targets, hand-keypoint targets, or both auxiliary targets. An ImageNet-only reference skipped the personal-video pretraining. Each visual model was then adapted using simulated images with known object positions.

With 256 adaptation images, all five groups succeeded in every recorded test run. With 32 images, failures appeared, but combined hand/object supervision did not improve success. This exposed the limit of the initial question: a strong fixed controller and an easy task could not tell us whether the model had learned to control a robot.

## Explore prediction and learned actions

A supporting experiment learned to predict the result of a complete grasp skill from the initial state and a proposed grasp command. Its endpoint predictions beat simple displacement baselines. This explored action-conditioned state prediction; it did not choose commands or plan the four-goal policy's actions.

A compact Diffusion Policy then learned robot command sequences from 128 additional simulated demonstrations. Its 84/192 strict successes showed an executable learned-control path, while failures made clear that fitting demonstration actions does not guarantee reliable physical execution.

## Make the destination explicit

The task expanded to four source positions, P1–P4, and four target centers, G1–G4. The required goal is chosen before a trial. Success now requires lifting, releasing, withdrawing and stable placement of the entire block inside a 40 mm square.

The recurrent policy uses the current image, robot state, goal and previous command. Development explored action representation, history, coordinate precision and color augmentation. The reported models and protocols were frozen before their respective test runs; the development sequence is not a matched comparison between architectures.

Training only P1→G1, P2→G2 and so on risks tying the grasp location to the destination. A coverage experiment therefore compared four same-number pairs with twelve pairs, holding the demonstration count at 24. Newly covered combinations gained some successes, but overall test success fell from 22/96 to 15/96. Coverage alone did not resolve the control problem.

## Test the personal video's additional contribution

The final study used eight preselected S01 clips and all sixteen X01 clips. Both conditions received identical video-derived target geometry and 24 robot demonstrations covering twelve pairs. Only one condition also used the clips' coarse transport time and image progress to shape teacher transport.

The same learner, paired initializations and sampling, three seeds and 2,500 updates were used in both conditions. The test produced 7/96 strict successes with shared geometry and 11/96 with added timing/progress. That difference is small, and most runs failed before a qualified lift.

The project establishes a traceable route from personal recordings to training demonstrations and learned simulation control. It leaves an open question: what data and training changes would make the learned grasp reliable enough for video-derived transport constraints to matter consistently?

![Research progression](../media/study-progression.svg)

Read the [method](METHOD.md) for the information flow and [experiments](EXPERIMENTS.md) for the exact comparisons and limits.
