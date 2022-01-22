#pragma once

/*
generate state: fill trace state
trace state: fill inct, splat le
bsdf state: sample light, generate shadow state, sample bsdf, fill trace state
shadow state: splat li
*/

/*
pipeline:
    while !done
        if active state count < threshold
            generate new state
        trace
        sort and modify active state count
        shade
        if active shadow state count > threshold
            shadow
*/
