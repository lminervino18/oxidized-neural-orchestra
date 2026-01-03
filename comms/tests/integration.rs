use comms::{Deserialize, Serialize};
use tokio::io;

struct MyStr<'a>(&'a str);

impl<'a> Serialize<'a> for MyStr<'_> {
    fn serialize(&'a self, _buf: &mut Vec<u8>) -> Option<&'a [u8]> {
        Some(self.0.as_bytes())
    }
}

impl<'a> Deserialize<'a> for MyStr<'a> {
    fn deserialize(buf: &'a [u8]) -> std::io::Result<Self> {
        Ok(Self(str::from_utf8(buf).unwrap()))
    }
}

#[test]
fn serialize_deserialize() {
    let s = MyStr("Hello, world!");
    let serialized = s.serialize(&mut Vec::new()).unwrap();
    let deserialized = MyStr::deserialize(serialized).unwrap();
    assert_eq!(deserialized.0, s.0);
}

#[tokio::test]
async fn send_recv() {
    const SIZE: usize = 128;

    let msg = MyStr("Hello, world!");

    let (one, two) = io::duplex(SIZE);
    let (rx, tx) = io::split(one);
    let (_, mut tx) = comms::channel(rx, tx);

    tx.send(&msg).await.unwrap();

    let (rx, tx) = io::split(two);
    let (mut rx, _) = comms::channel(rx, tx);

    let s: MyStr = rx.recv().await.unwrap();

    assert_eq!(msg.0, s.0);
}
