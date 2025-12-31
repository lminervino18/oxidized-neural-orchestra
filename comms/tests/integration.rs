use std::io::{self, Read, Write};

use bytes::{Buf, BufMut, BytesMut};

use comms::{Deserialize, Serialize};

struct MyString(String);

impl Serialize for MyString {
    fn serialize<B: BufMut>(&self, buf: &mut B) -> io::Result<()> {
        let mut writer = buf.writer();
        writer.write_all(self.0.as_bytes())
    }
}

impl<'a> Deserialize<'a> for MyString {
    fn deserialize<B: Buf>(buf: &'a mut B) -> io::Result<Self> {
        let mut s = Vec::new();
        let mut reader = buf.reader();
        reader.read_to_end(&mut s)?;
        Ok(MyString(String::from_utf8(s).unwrap()))
    }
}

#[test]
fn test_serialization_deserialization() {
    let mut buf = BytesMut::new();

    let s = MyString("Hello, world!".to_string());
    s.serialize(&mut buf).unwrap();

    let res = MyString::deserialize(&mut buf).unwrap();
    assert_eq!(s.0, res.0);
}
