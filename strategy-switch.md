# Notes

1. Hay que agregar una manera de medir cuando se va a efectuar el switch y una manera de ejecutarlo (seria mandar los
   mensajes ya definidos en `comms/src/protocol/msg.rs`).
2. Hay que leer bien cual es la regla del switch, una vez que tenga esto implementado deberia estar listo para probar
   el algoritmo.
3. Puedo usar el ConvergenceTracker para esto, estaria bueno tener una ventana mas grande, se ve que en el paper es de
   las 5 losses anteriores, una vez que alcanza o supera un `s` entonces se triggerea el switch.

Armar un loss recorder para ir registrando las losses de los workers, una vez que tengo la loss creada, le puedo pedir la mean, max, etc.
Luego esa loss se la doy al switch_tracker.
